# -*- coding: utf-8 -*-

import json
import pandas as pd
from tqdm import tqdm
import argparse
import os
from hashlib import blake2b
import traceback
import time
from collections import defaultdict
import re

NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"

# ==== 新 SFT 标记 ====
PSEP_TOKEN = "<|psep|>"
PSEP_BLOCK_HEADER = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_FOOTER = "\n</POTENTIAL_SPANS>\n"
CSTART_TOKEN = "<|cstart|>"
CEND_TOKEN   = "<|cend|>"

_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$",
                               re.IGNORECASE | re.MULTILINE)

def _extract_parent_name_from_user_content(user_content_str: str) -> str:
    if not isinstance(user_content_str, str):
        return ""
    m = _USERNAME_LINE_RE.search(user_content_str)
    return (m.group("name") or "").strip() if m else ""

# ---------- 基础工具 ----------
def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _as_py(x): return x.as_py() if hasattr(x, "as_py") else x

def _normalize_messages(cell):
    if cell is None: return []
    if isinstance(cell, str):
        try:
            obj = json.loads(cell); return obj if isinstance(obj, list) else []
        except Exception: return []
    try: it = list(cell)
    except TypeError:
        one = _as_py(cell); return [one] if isinstance(one, dict) else []
    return [_as_py(m) for m in it]

def _normalize_system(msg):
    if not msg or msg.get("role") != "system": return None
    content = msg.get("content", "")
    try:
        j = json.loads(content)
        if isinstance(j, str): content = j
    except Exception: pass
    return {"role": "system", "content": content}

def _hash_prompt(system_content: str, root_user_str: str) -> str:
    h = blake2b(digest_size=16)
    h.update((system_content or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((root_user_str or "").encode("utf-8"))
    return h.hexdigest()

# ---------- 旧格式兜底 ----------
def _parse_children_names_from_assistant_old(gt_text: str) -> list[str]:
    s = (gt_text or "").strip()
    if not s or s == NO_INTERACTION_STR or s == "[]": return []
    if s.startswith("[INTERACT="):
        pos_rn = s.find("\r\n"); pos_n = s.find("\n")
        pos = pos_rn if (pos_rn != -1 and (pos_n == -1 or pos_rn < pos_n)) else pos_n
        nl_len = 2 if pos == pos_rn and pos_rn != -1 else (1 if pos != -1 else 0)
        if pos == -1:
            return []
        s = s[pos + nl_len:].strip()
    try: j = json.loads(s)
    except Exception: j = None
    out = []
    if isinstance(j, list):
        for it in j:
            if isinstance(it, dict):
                name = (it.get("user_name") or "").strip()
                if name: out.append(name)
            elif isinstance(it, str) and it.strip():
                out.append(it.strip())
        return _dedup_keep_order(out)
    if isinstance(j, dict):
        name = (j.get("user_name") or "").strip()
        return [name] if name else []
    for sep in [",", "，", "、", ";", "；", "|", " "]:
        if sep in s:
            return [p for p in (pp.strip() for pp in s.split(sep)) if p]
    return [s]

# ---------- SFT: POTENTIAL_SPANS 解析 ----------
def _iter_potential_blocks_from_content(user_content_str: str):
    s = user_content_str or ""
    hdr_pos = s.rfind(PSEP_BLOCK_HEADER)
    if hdr_pos < 0: return
    tail = s[hdr_pos + len(PSEP_BLOCK_HEADER):]
    end_pos = tail.find(PSEP_BLOCK_FOOTER)
    if end_pos != -1: tail = tail[:end_pos]
    parts = tail.split(PSEP_TOKEN)
    for part in parts:
        part = part.strip()
        if not part or part == PSEP_TOKEN: continue
        try:
            blk = json.loads(part)
            if isinstance(blk, dict): yield blk
        except Exception:
            continue

def _extract_node_depth_from_user_content(user_content_str: str) -> int:
    for blk in _iter_potential_blocks_from_content(user_content_str):
        try: return int(blk.get("depth", 0))
        except Exception: return 0
    return 0

def _candidate_order_from_user_content(user_content_str: str) -> list[str]:
    order = []
    for blk in _iter_potential_blocks_from_content(user_content_str):
        n = str(blk.get("user_name") or "").strip()
        if n: order.append(n)
    return order

def _extract_root_potential_full(user_content_str: str) -> list[dict]:
    pots = []
    for blk in _iter_potential_blocks_from_content(user_content_str):
        if not isinstance(blk, dict): continue
        obj = {
            "user_name": str(blk.get("user_name") or "").strip(),
            "interests": blk.get("interests") or [],
            "depth": int(blk.get("depth", 0)) if isinstance(blk.get("depth", 0), (int, float, str)) else 0,
        }
        if obj["user_name"]:
            pots.append(obj)
    return pots

# ---------- 新 SFT: assistant 覆盖数组解析 ----------
def _parse_positive_children_from_new_assistant(asst_text: str, candidate_order: list[str]) -> list[str]:
    s = (asst_text or "").strip()
    try: arr = json.loads(s)
    except Exception: arr = None
    if isinstance(arr, list) and (len(arr) == 0 or (isinstance(arr[0], dict) and "type" in arr[0])):
        types, names = [], []
        for it in arr:
            try: t = int(it.get("type", 0))
            except Exception: t = 0
            types.append(t)
            nm = (it.get("user_name") or "").strip()
            names.append(nm if nm else None)
        if candidate_order and (any(n is None for n in names) or len(names) != len(candidate_order)):
            m = min(len(types), len(candidate_order))
            names = candidate_order[:m]; types = types[:m]
        seen, out = set(), []
        for nm, t in zip(names, types):
            if nm and (t in (1, 2)) and nm not in seen:
                seen.add(nm); out.append(nm)
        return out
    return _dedup_keep_order(_parse_children_names_from_assistant_old(s))

# ---------- 分片信息 ----------
_META_ROLE_KEY = "meta"
_ID_RE = re.compile(r"^(?P<rec>.+?)_root_(?P<root>.+?)_flchunk_(?P<idx>\d+)$")

def _extract_meta_from_messages(messages: list[dict]) -> dict | None:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == _META_ROLE_KEY:
            c = m.get("content", "")
            try:
                j = json.loads(c) if isinstance(c, str) else (c if isinstance(c, dict) else None)
                if isinstance(j, dict): return j
            except Exception: return None
    return None

def _parse_chunk_from_sample_id(sample_id: str) -> dict:
    out = {"sample_id": sample_id or None, "record_id": None, "root_user": None, "chunk_index": None}
    if not sample_id: return out
    m = _ID_RE.match(sample_id)
    if m:
        out["record_id"] = m.group("rec"); out["root_user"] = m.group("root")
        try: out["chunk_index"] = int(m.group("idx"))
        except Exception: out["chunk_index"] = None
    return out

def _compose_sft_chunk_info(row: pd.Series, messages: list[dict]) -> dict:
    meta = _extract_meta_from_messages(messages) or {}
    sample_id = row.get("id") if isinstance(row, dict) else (row.id if hasattr(row, "id") else None)
    if sample_id is None:
        try: sample_id = row["id"]
        except Exception: sample_id = None
    parsed = _parse_chunk_from_sample_id(str(sample_id) if sample_id is not None else "")
    return {
        "sample_id": parsed.get("sample_id"),
        "record_id": meta.get("record_id", parsed.get("record_id")),
        "root_user": meta.get("root_user", parsed.get("root_user")),
        "chunk_index": meta.get("chunk_index", parsed.get("chunk_index")),
        "chunk_size": meta.get("chunk_size"),
        "first_layer_users": meta.get("first_layer_users"),
    }

# ---------- 写 Parquet ----------
def _safe_to_parquet(dff: pd.DataFrame, path: str, parquet_engine: str, parquet_compression: str):
    attempts = [
        (parquet_engine, parquet_compression),
        ("pyarrow", "zstd"),
        ("pyarrow", None),
        ("fastparquet", None),
    ]
    last_err = None
    for eng, comp in attempts:
        try:
            print(f"[parquet] writing -> {path} (engine={eng}, compression={comp})")
            t1 = time.time()
            dff.to_parquet(path, index=False, engine=eng, compression=comp)
            print(f"[parquet] done: {path}  rows={len(dff)}  took={time.time()-t1:.2f}s")
            return
        except Exception as e:
            last_err = e
            print(f"[parquet][warn] engine={eng}, compression={comp} failed: {e}")
            traceback.print_exc()
            continue
    if last_err is not None: raise last_err

# ---------- 构造 cond_gt_by_turn ----------
def _build_strict_parent_cond_gt(pairs: list[tuple[dict, dict]], *, enforce_children_in_potential: bool=False) -> list[list[dict]]:
    depth_parent2children: dict[int, dict[str, list[str]]] = defaultdict(dict)
    observed_depths = set()
    for (u, a) in pairs:
        u_content = u.get("content", "")
        parent_name = _extract_parent_name_from_user_content(u_content) or "__ROOT__"
        d = _extract_node_depth_from_user_content(u_content); observed_depths.add(d)
        cand_order = _candidate_order_from_user_content(u_content)
        child_names = _parse_positive_children_from_new_assistant(a.get("content", "") or "", candidate_order=cand_order)
        if enforce_children_in_potential and cand_order and child_names:
            cand_set = set(cand_order); child_names = [c for c in child_names if c in cand_set]
        if parent_name not in depth_parent2children[d]:
            depth_parent2children[d][parent_name] = []
        if child_names:
            existed = depth_parent2children[d][parent_name]; exist_set = set(existed)
            for c in child_names:
                if c not in exist_set:
                    existed.append(c); exist_set.add(c)
    if not observed_depths: return []
    min_d, max_d = min(observed_depths), max(observed_depths)
    L = max_d - min_d + 1
    cond_gt_by_turn: list[list[dict]] = [[] for _ in range(L)]
    for d in range(min_d, max_d + 1):
        rd = d - min_d; p2c = depth_parent2children.get(d, {})
        if not p2c: cond_gt_by_turn[rd] = []; continue
        cond_gt_by_turn[rd] = [{parent: list(children)} for parent, children in p2c.items()]
    return cond_gt_by_turn

# ---------- 单文件（val） ----------
def _build_grpo_for_one_file(
    sft_parquet_path: str,
    out_parquet: str,
    *,
    data_source: str = "social_f1",
    seed: int = 42,
    dedup_by_prompt: bool = True,
    jsonl_out: str | None = None,
    parquet_engine: str = "pyarrow",
    parquet_compression: str = "snappy",
    max_prompt_len: int = 4096,
    num_proc: int = 4,
    keep_all_no_interact: bool = True,
    allow_system_anywhere: bool = True,
    enforce_children_in_potential: bool = False,
    embed_root_potential_full: bool = True,
):
    t0 = time.time()
    print(f"[load] reading sft parquet: {sft_parquet_path}")
    df = pd.read_parquet(sft_parquet_path)
    print(f"[load] rows: {len(df)}")

    samples = []
    seen = set()
    kept, drop_no_system, drop_no_root, drop_no_pairs = 0, 0, 0, 0

    for ridx, row in tqdm(df.iterrows(), total=len(df), desc=f"Build GRPO ({os.path.basename(sft_parquet_path)})"):
        messages = _normalize_messages(row.get("messages"))
        if not isinstance(messages, list) or len(messages) < 2:
            drop_no_root += 1; continue

        system_msg = None
        if allow_system_anywhere:
            for mi in messages:
                if isinstance(mi, dict) and mi.get("role") == "system":
                    system_msg = _normalize_system(mi); break
        else:
            if isinstance(messages[0], dict) and messages[0].get("role") == "system":
                system_msg = _normalize_system(messages[0])
        sys_content = system_msg["content"] if system_msg else ""

        root_idx = None
        for i in range(0, len(messages)):
            mi = messages[i]
            if isinstance(mi, dict) and mi.get("role") == "user":
                root_idx = i; break
        if root_idx is None:
            drop_no_root += 1; continue

        root_user_raw = messages[root_idx].get("content", "") or ""
        try: root_user_json = json.loads(root_user_raw) if isinstance(root_user_raw, str) else None
        except Exception: root_user_json = None
        if not isinstance(root_user_json, dict):
            root_user_json = {"content": root_user_raw}

        pairs = []
        i = root_idx
        while i + 1 < len(messages):
            u, a = messages[i], messages[i + 1]; i += 2
            if not (isinstance(u, dict) and isinstance(a, dict)): continue
            if u.get("role") != "user" or a.get("role") != "assistant": continue
            pairs.append((u, a))
        if len(pairs) == 0:
            drop_no_pairs += 1; continue

        cond_gt_by_turn = _build_strict_parent_cond_gt(pairs, enforce_children_in_potential=enforce_children_in_potential)
        any_child = any(len(layer) > 0 and any(len(next(iter(d.values()))) > 0 for d in layer) for layer in cond_gt_by_turn)
        if (not keep_all_no_interact) and (not any_child): continue

        # 根池解析 + 父键
        root_content_text = root_user_json.get("content", "") or ""
        root_pots_full = _extract_root_potential_full(root_content_text)
        root_pots_names = [p["user_name"] for p in root_pots_full]
        has_psep_block = len(root_pots_full) > 0
        dmin = min((p.get("depth", 0) for p in root_pots_full), default=0) if has_psep_block else 0
        dmax = max((p.get("depth", 0) for p in root_pots_full), default=0) if has_psep_block else 0
        root_parent_key = _extract_parent_name_from_user_content(root_content_text) or "__ROOT__"

        # 写入 reward_model（评估时会剥离）
        root_user_json.setdefault("historical_interactors", [])
        root_user_json["reward_model"] = {
            "ground_truth": {"cond_gt_by_turn": cond_gt_by_turn},
            "root_potential": {
                "user_names": root_pots_names,
                **({"full": root_pots_full} if embed_root_potential_full else {}),
            },
            "root_parent_key": root_parent_key,
        }

        root_user_str = json.dumps(root_user_json, ensure_ascii=False)
        prompt_msgs = [{"role": "system", "content": sys_content},
                       {"role": "user", "content": root_user_str}]

        sft_chunk_info = _compose_sft_chunk_info(row, messages) or {}
        sft_chunk_info = dict(sft_chunk_info)
        sft_chunk_info.update({
            "root_potential_count": len(root_pots_full),
            "root_potential_min_depth": int(dmin),
            "root_potential_max_depth": int(dmax),
            "has_psep_block": bool(has_psep_block),
        })

        if dedup_by_prompt:
            key = _hash_prompt(sys_content, root_user_str)
            if key in seen: continue
            seen.add(key)

        samples.append({
            "data_source": data_source,
            "prompt":      prompt_msgs,
            "ability":     "social_prediction",
            "sft_chunk_info": sft_chunk_info,
        })
        kept += 1

    print("========== Build Summary ==========")
    print(f"[file] {sft_parquet_path}")
    print(f"Total dialogs            : {len(df)}")
    print(f"Kept (1 row per dialog)  : {kept}")
    print(f"Dropped no root          : {drop_no_root}")
    print(f"Dropped no (u,a) pairs   : {drop_no_pairs}")

    if jsonl_out:
        os.makedirs(os.path.dirname(jsonl_out) or ".", exist_ok=True)
        print(f"[write] writing JSONL -> {jsonl_out}")
        with open(jsonl_out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[write] JSONL done: {len(samples)} lines")

    print("[parquet] prepare dataframe")
    df_grpo = pd.DataFrame(
        {
            "data_source":     [s["data_source"] for s in samples],
            "prompt":          [s["prompt"] for s in samples],
            "ability":         [s["ability"] for s in samples],
            "sft_chunk_info":  [s["sft_chunk_info"] for s in samples],
        }
    )
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    _safe_to_parquet(df_grpo, out_parquet, parquet_engine, parquet_compression)
    print(f"[done] file completed in {time.time()-t0:.2f}s")

# ---------- 主入口（仅 val） ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build GRPO dataset (VAL only) — read NEW SFT (with <POTENTIAL_SPANS> & coverage array 0/1/2); output reward_model.ground_truth.cond_gt_by_turn and root_potential meta."
    )
    # 只需要 val 输入/输出
    ap.add_argument("--val_sft_parquet",   required=True, help="验证集 SFT parquet")
    ap.add_argument("--val_output",        required=True, help="验证集 GRPO parquet 输出路径")
    ap.add_argument("--val_jsonl_output",  default=None,  help="可选：验证集 JSONL 输出")

    # 其余参数
    ap.add_argument("--data_source", default="social_f1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_dedup", action="store_true", help="disable prompt deduplication")
    ap.add_argument("--parquet_engine", default="pyarrow", choices=["pyarrow", "fastparquet"])
    ap.add_argument("--parquet_compression", default="snappy")
    ap.add_argument("--max_prompt_len", type=int, default=4096)
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--drop_all_no_interact", action="store_true",
                    help="if set, will drop dialogs whose all turns are empty (default: keep)")
    ap.add_argument("--enforce_children_in_potential", action="store_true",
                    help="若设置，则把每层 children 过滤为该步候选顺序子集（按 user_name 匹配）。")
    ap.add_argument("--embed_root_potential_full", action="store_true", default=True,
                    help="在 reward_model 中包含 root_potential.full（完整对象数组）。默认开启。")

    args = ap.parse_args()

    _build_grpo_for_one_file(
        sft_parquet_path=args.val_sft_parquet,
        out_parquet=args.val_output,
        jsonl_out=args.val_jsonl_output,
        data_source=args.data_source,
        seed=args.seed,
        dedup_by_prompt=(not args.no_dedup),
        parquet_engine=args.parquet_engine,
        parquet_compression=args.parquet_compression,
        max_prompt_len=args.max_prompt_len,
        num_proc=args.num_proc,
        keep_all_no_interact=(not args.drop_all_no_interact),
        enforce_children_in_potential=args.enforce_children_in_potential,
        embed_root_potential_full=bool(args.embed_root_potential_full),
    )
