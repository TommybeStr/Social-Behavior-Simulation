#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, ast, argparse
from typing import List, Dict, Tuple, Any, Optional, Iterable
from collections import Counter, defaultdict

NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
VALID_TYPES = {"评论", "转发", "转发微博"}

# 允许写回的字段
KEEP_DETAIL_KEYS = {
    "ts","group_id","trajectory_id","depth","report_depth","cond_key",
    "input_user","input_text","output_text","gold","parse_status",
    "finish_reason"
}

# -------------------- 基础工具 --------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def dedup_keep_order(seq: Iterable[Any]):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# -------------------- 模板/围栏清除 --------------------
_CHAT_ARTIFACTS_PATTERNS = [
    r'^\s*<\|im_start\|>\s*assistant\s*\r?\n?',
    r'<\|im_end\|>\s*$',
    r'^\s*<\|im_start\|>\s*user\s*\r?\n?',
    r'^\s*\[user.*?\]\s*\r?\n?',
]

def strip_chat_artifacts(s: str) -> str:
    t = s
    t = re.sub(r"^\s*```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t, flags=re.IGNORECASE)
    for pat in _CHAT_ARTIFACTS_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.MULTILINE)
    return t.strip()

# -------------------- 宽松标签头剥离 --------------------
_INTERACT_HEAD_BRACKET = re.compile(
    r'^\s*\[\s*[^\]\n\r]*?INTERACT\s*[:=]\s*([01])\s*[^\]\n\r]*\]\s*(?:\r?\n)?',
    re.IGNORECASE
)
_INTERACT_HEAD_NOBRACKET = re.compile(
    r'^\s*INTERACT\s*[:=]\s*([01])\s*(?:\r?\n|$)',
    re.IGNORECASE
)

def peel_interact_header_loose(s: str) -> Tuple[Optional[int], str, Optional[str]]:
    if not isinstance(s, str):
        return None, "", None
    m = _INTERACT_HEAD_BRACKET.match(s)
    if m:
        return int(m.group(1)), s[m.end():].strip(), "peeled_header_bracket"
    m2 = _INTERACT_HEAD_NOBRACKET.match(s)
    if m2:
        return int(m2.group(1)), s[m2.end():].strip(), "peeled_header_nobracket"
    return None, s.strip(), None

# -------------------- JSON 片段提取与修复 --------------------
def _fix_trailing_commas(txt: str) -> str:
    return re.sub(r",\s*(\]|\})", r"\1", txt)

def _maybe_wrap_as_list(txt: str) -> str:
    stripped = txt.strip()
    if stripped.startswith("["):
        return txt
    if "[" not in stripped and "]" not in stripped and "{" in stripped and "}" in stripped:
        return f"[{txt}]"
    return txt

def _balanced_json_array_slice(s: str) -> Optional[str]:
    start = s.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    if depth > 0:
        return s[start:] + ("]" * min(depth, 5))
    return None

def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _try_literal_eval(s: str) -> Optional[Any]:
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def extract_and_repair_list(s: str) -> Tuple[Optional[List[Any]], str]:
    txt0 = s.strip()
    j = _try_json_loads(txt0)
    if isinstance(j, list):
        return j, "direct_json"

    frag = _balanced_json_array_slice(txt0)
    if frag:
        j = _try_json_loads(frag)
        if isinstance(j, list):
            return j, "slice_json_array"
        frag2 = _fix_trailing_commas(frag)
        j = _try_json_loads(frag2)
        if isinstance(j, list):
            return j, "slice_json_array_fix_commas"

    txt1 = _maybe_wrap_as_list(txt0)
    if txt1 != txt0:
        j = _try_json_loads(txt1)
        if isinstance(j, list):
            return j, "wrapped_as_list"
        j = _try_literal_eval(txt1)
        if isinstance(j, list):
            return j, "wrapped_literal_eval"

    txt2 = _fix_trailing_commas(txt0)
    j = _try_json_loads(txt2)
    if isinstance(j, list):
        return j, "fix_commas_only"
    j = _try_literal_eval(txt2)
    if isinstance(j, list):
        return j, "literal_eval_fix_commas"

    j = _try_literal_eval(txt0)
    if isinstance(j, list):
        return j, "literal_eval_raw"

    return None, "fail"

# -------------------- 业务校验/规范化 --------------------
def to_valid_items(lst: List[Any]) -> Tuple[List[Dict[str, str]], str]:
    out = []
    if not isinstance(lst, list):
        return [], "not_list"

    for it in lst:
        if not isinstance(it, dict):
            return [], "item_not_dict"
        u, c, t = it.get("user_name"), it.get("content"), it.get("type")
        if not isinstance(u, str) or not isinstance(c, str) or not isinstance(t, str):
            return [], "field_not_str"
        u, c, t = u.strip(), c.strip(), t.strip()
        if not u:
            continue
        if t == "转发":
            t = "转发微博"
        if t not in VALID_TYPES:
            return [], "bad_type"
        out.append({"user_name": u, "content": c, "type": t})

    seen, uniq = set(), []
    for x in out:
        n = x.get("user_name")
        if n and n not in seen:
            seen.add(n); uniq.append(x)
    return uniq, "ok" if len(uniq) > 0 else "empty"

# -------------------- 主清洗：针对一条 output_text --------------------
def clean_one_output(raw_text: str) -> Tuple[List[Dict[str, str]], str, str, bool]:
    s = (raw_text or "").strip()
    if not s:
        return [], "parse_fail_empty", "empty_output", False

    s = strip_chat_artifacts(s)

    tag, body, note1 = peel_interact_header_loose(s)
    if tag == 0:
        return [], "no_interaction", note1 or "tag0", True
    if tag == 1:
        s = body

    if s.strip() in ("[]", NO_INTERACTION_STR):
        return [], "empty", "explicit_empty", True

    arr, note2 = extract_and_repair_list(s)
    if arr is None:
        return [], "parse_fail_json", note2, False

    items, vflag = to_valid_items(arr)
    if vflag == "ok":
        return items, "ok_repaired", (note1 or "") + (";" if note1 else "") + note2, True
    if vflag == "empty":
        return [], "empty", (note1 or "") + (";" if note1 else "") + note2 + ";empty_after_filter", True

    status_map = {
        "not_list": "parse_fail_not_list",
        "item_not_dict": "parse_fail_item_not_dict",
        "field_not_str": "parse_fail_field_not_str",
        "bad_type": "parse_fail_bad_field",
    }
    return [], status_map.get(vflag, "parse_fail_bad_field"), (note1 or "") + (";" if note1 else "") + note2 + f";{vflag}", False

# -------------------- CLI：读取 detail JSONL，输出修复版 + 统计 --------------------
def main():
    ap = argparse.ArgumentParser(description="Clean & recover model outputs from detail JSONL with per-depth stats.")
    ap.add_argument("--input", required=True, help="detail JSONL（例如：rollout_io_gold.jsonl）")
    ap.add_argument("--output", required=True, help="清洗后的 JSONL 输出路径")
    ap.add_argument("--summary", default="recovery_summary.txt", help="统计汇总输出路径")
    args = ap.parse_args()

    ensure_dir(args.output); ensure_dir(args.summary)
    totals = Counter()
    repaired = Counter()

    # 新增：按 depth 统计（清洗后）
    # stats_by_depth[depth] = {
    #   "has_gold": x, "no_gold": y, "has_pred": a, "no_pred": b
    # }
    stats_by_depth = defaultdict(lambda: {"has_gold": 0, "no_gold": 0, "has_pred": 0, "no_pred": 0})

    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in load_jsonl(args.input):
            base = {k: rec.get(k) for k in KEEP_DETAIL_KEYS if k in rec}
            raw = rec.get("output_text", "")
            gold_list = rec.get("gold") or []
            depth = int(rec.get("depth", 0) or 0)

            parsed, status_rec, note, ok = clean_one_output(raw)

            base["parsed_output_recovered"] = parsed
            base["parse_status_recovered"] = status_rec
            base["recovery_note"] = note
            base["recovered"] = bool(ok)

            totals["all"] += 1
            totals[rec.get("parse_status","unknown")] += 1
            repaired[status_rec] += 1
            if ok: totals["recovered_ok"] += 1
            else:   totals["recovered_fail"] += 1

            # ---- 分层统计（清洗后）----
            has_gold = isinstance(gold_list, list) and len(gold_list) > 0
            has_pred = isinstance(parsed, list) and len(parsed) > 0
            if has_gold: stats_by_depth[depth]["has_gold"] += 1
            else:        stats_by_depth[depth]["no_gold"]  += 1
            if has_pred: stats_by_depth[depth]["has_pred"] += 1
            else:        stats_by_depth[depth]["no_pred"]  += 1

            fout.write(json_dump(base) + "\n")

    # ---- 汇总写入 ----
    with open(args.summary, "w", encoding="utf-8") as fw:
        fw.write("=== Input parse_status distribution ===\n")
        for k, v in totals.items():
            if k in ("all","recovered_ok","recovered_fail"):
                continue
            fw.write(f"{k}: {v}\n")

        fw.write("\n=== Recovered parse_status distribution ===\n")
        for k, v in repaired.items():
            fw.write(f"{k}: {v}\n")

        fw.write("\n=== Overall ===\n")
        fw.write(f"total: {totals['all']}\n")
        fw.write(f"recovered_ok: {totals['recovered_ok']}\n")
        fw.write(f"recovered_fail: {totals['recovered_fail']}\n")

        fw.write("\n=== Per-depth post-cleaning stats ===\n")
        fw.write("(depth 使用 0 基；has_gold/no_gold 取自 gold 列；has_pred/no_pred 基于清洗后的 parsed_output_recovered)\n")
        for d in sorted(stats_by_depth.keys()):
            st = stats_by_depth[d]
            fw.write(f"depth={d}: has_gold={st['has_gold']}, no_gold={st['no_gold']}, "
                     f"has_pred={st['has_pred']}, no_pred={st['no_pred']}\n")

    print(f"[OK] cleaned JSONL -> {args.output}")
    print(f"[OK] summary       -> {args.summary}")

if __name__ == "__main__":
    main()
