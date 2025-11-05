# -*- coding: utf-8 -*-
import json
import ast
import argparse
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 常量 =====
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
VALID_TYPES = {"评论", "转发", "转发微博"}

SYSTEM_PROMPT = (
    "你是一个社交媒体互动预测专家，能够根据输入博文的具体内容，预测该条博文的互动情况。"
    "输入字段包括：- user_name - user_interests - content - depth - historical_interactors - potential_interactors（你只能从中选择用户进行预测）。"
    "你必须严格按照以下格式输出：[{\"user_name\": \"用户名（来自potential_interactors）\", \"content\": \"预测的评论内容\", \"type\": \"评论 或 转发\"}, ...] "
    "注意事项：1. 只能从 potential_interactors 中选择用户；"
    "2. type 只能为 \"评论\" 或 \"转发\"；"
    "3. 不允许输出任何解释、分析、说明；"
    "4. 如果你认为该帖子没有任何互动，输出一个空数组 []。"
)

# ---------------- 状态工具 ---------------- #
def is_parse_fail(status: str) -> bool:
    return isinstance(status, str) and status.startswith("parse_fail")

def canonical_parse_status(old: str) -> str:
    mapping = {
        "parse_fail": "parse_fail_bad_field",
        "parse_fail_can't_load": "parse_fail_json",
        "parse_fail_no_list": "parse_fail_not_list",
        "parse_fail_no_dict": "parse_fail_item_not_dict",
        "parse_fail_nostr": "parse_fail_field_not_str",
        "parse_fail_empty": "parse_fail_empty",
        "parse_fail_json": "parse_fail_json",
        "parse_fail_not_list": "parse_fail_not_list",
        "parse_fail_item_not_dict": "parse_fail_item_not_dict",
        "parse_fail_field_not_str": "parse_fail_field_not_str",
        "parse_fail_bad_field": "parse_fail_bad_field",
        "ok": "ok",
        "ok_repaired": "ok_repaired",
        "empty": "empty",
        "no_interaction": "no_interaction",
    }
    return mapping.get(old, old if is_parse_fail(old) else old)

# ---------------- 更鲁棒的泛解析 ---------------- #
def smart_parse(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, (np.generic,)):
        x = x.item()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.tolist()
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return default
    return default

# ---------------- 半拉子 JSON 修复 ---------------- #
def _extract_array_span(s: str) -> str:
    if not s:
        return ""
    i0 = s.find('[')
    return s[i0:] if i0 != -1 else ""

def _repair_truncated_array(s: str) -> Optional[str]:
    """
    尝试把包含未闭合 JSON 数组的纯文本修复为可解析的 JSON：
    - 仅在文本中找到 '[' 后的部分进行处理
    - 利用括号/字符串状态追踪，判断最近一个完整对象位置
    - 截断尾部悬空对象/逗号，并补上右方括号
    """
    frag = _extract_array_span((s or "").strip())
    if not frag or frag[0] != '[':
        return None

    in_str = False
    esc = False
    depth = 0
    last_complete_obj_end = -1  # 记录最后一个 '}'（且处于 depth==1）的下标

    for i, ch in enumerate(frag):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                # 已经天然闭合，若可整体解析则直接返回
                try:
                    json.loads(frag[:i + 1])
                    return frag[:i + 1]
                except Exception:
                    break
        elif ch == '}':
            if depth == 1:
                last_complete_obj_end = i

    if last_complete_obj_end != -1:
        cut = frag[:last_complete_obj_end + 1].rstrip()
        if cut.endswith(','):
            cut = cut[:-1].rstrip()
        candidate = cut + ']'
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None

# ---------------- 解析 & 校验（支持修复） ---------------- #
def parse_and_validate(model_resp: str) -> Tuple[List[Dict[str, str]], str, Optional[str]]:
    """
    返回: (preds, status, repaired_output)
    - 正常解析成功 -> status="ok"
    - 修复后解析成功 -> status="ok_repaired"，repaired_output 返回修复后的 JSON 字符串
    - 其它失败 -> status=各种 parse_fail_*
    """
    s = (model_resp or "").strip()
    if not s:
        return [], "parse_fail_empty", None
    if s == NO_INTERACTION_STR:
        return [], "no_interaction", None

    repaired_output = None
    try:
        parsed = json.loads(s)
    except Exception:
        # 尝试修复未闭合数组
        repaired = _repair_truncated_array(s)
        if not repaired:
            return [], "parse_fail_json", None
        try:
            parsed = json.loads(repaired)
            repaired_output = repaired
            status_now = "ok_repaired"
        except Exception:
            return [], "parse_fail_json", None
    else:
        status_now = "ok"

    if not isinstance(parsed, list):
        return [], "parse_fail_not_list", repaired_output

    out = []
    for it in parsed:
        if not isinstance(it, dict):
            return [], "parse_fail_item_not_dict", repaired_output
        u = it.get("user_name")
        c = it.get("content")
        t = it.get("type")
        if not isinstance(u, str) or not isinstance(c, str) or not isinstance(t, str):
            return [], "parse_fail_field_not_str", repaired_output
        u = u.strip(); c = c.strip(); t = t.strip()
        if (not u) or (t not in VALID_TYPES):
            return [], "parse_fail_bad_field", repaired_output
        out.append({"user_name": u, "content": c, "type": t})

    # 去重（按 user_name 保序）
    seen, uniq = set(), []
    for x in out:
        if x["user_name"] not in seen:
            seen.add(x["user_name"])
            uniq.append(x)

    if len(uniq) == 0:
        return [], "empty", repaired_output
    return uniq, status_now, repaired_output

# ---------------- GT 展开与 F1 ---------------- #
def to_names_from_any_gold(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s == NO_INTERACTION_STR:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return to_names_from_any_gold(parsed)
        except Exception:
            pass
        return [s]
    out: List[str] = []
    if isinstance(x, dict):
        n = (x.get("user_name") or "").strip()
        return [n] if n else []
    if isinstance(x, list):
        for it in x:
            if isinstance(it, str):
                s = it.strip()
                if not s or s == NO_INTERACTION_STR:
                    continue
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        out.extend(to_names_from_any_gold(parsed))
                    else:
                        out.append(s)
                except Exception:
                    out.append(s)
            elif isinstance(it, dict):
                n = (it.get("user_name") or "").strip()
                if n:
                    out.append(n)
    # 去重保序
    seen, uniq = set(), []
    for n in out:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def set_f1(pred_names: List[str], gold_names: List[str]) -> float:
    pset = set([n for n in pred_names if isinstance(n, str) and len(n) > 0])
    gset = set([n for n in gold_names if isinstance(n, str) and len(n) > 0])
    if len(pset) == 0 and len(gset) == 0:
        return 1.0
    if len(pset) == 0 or len(gset) == 0:
        return 0.0
    inter = len(pset & gset)
    prec = inter / (len(pset) + 1e-6)
    rec = inter / (len(gset) + 1e-6)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-6)

# ---------------- 从数据行抽 root/gt ---------------- #
def get_root_user_json(row: pd.Series) -> Dict[str, Any]:
    raw = row.get("prompt", None)
    if raw is None and "messages" in row:
        raw = row["messages"]

    obj = smart_parse(raw, default=raw)
    if isinstance(obj, dict):
        if "prompt" in obj:
            obj = smart_parse(obj["prompt"], default=obj["prompt"])
        elif "messages" in obj:
            obj = smart_parse(obj["messages"], default=obj["messages"])

    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("prompt 字段无法解析为 list")

    first_user_idx = None
    for i, m in enumerate(obj):
        if isinstance(m, dict) and m.get("role") == "user":
            first_user_idx = i
            break
    if first_user_idx is None:
        raise ValueError("在 prompt 列表中未找到 user")

    root_user_json = smart_parse(obj[first_user_idx].get("content"), default=None)
    if not isinstance(root_user_json, dict):
        raise ValueError("root user content 不是 JSON 对象")
    return root_user_json

def get_gt_by_turn(row: pd.Series) -> List[List[str]]:
    rm = row.get("reward_model", None)
    rm_obj = smart_parse(rm, default=rm)

    gt_blob = None
    if isinstance(rm_obj, dict):
        gt_blob = rm_obj.get("ground_truth", None)
    if gt_blob is None:
        gt_blob = row.get("reward_model.ground_truth", None)

    gt = smart_parse(gt_blob, default=None)
    if not isinstance(gt, dict) or "gt_by_turn" not in gt:
        raise ValueError("ground_truth 缺少 gt_by_turn")

    arr = gt["gt_by_turn"]
    out: List[List[str]] = []
    for x in arr:
        out.append(to_names_from_any_gold(x))
    return out

def pot_list_to_map(pot_list) -> Dict[str, Dict[str, Any]]:
    m = {}
    for p in pot_list or []:
        if isinstance(p, dict):
            name = (p.get("user_name") or "").strip()
            if name:
                m[name] = p
        else:
            name = str(p).strip()
            if name:
                m[name] = {"user_name": name}
    return m

# ---------------- 单步生成 ---------------- #
@torch.no_grad()
def generate_one_step(model, tokenizer, device, user_json: Dict[str, Any], max_new_tokens: int, temperature: float):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_json, ensure_ascii=False)},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_text], return_tensors="pt").to(device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.9))

    gen_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **gen_kwargs
    )
    new_tokens = gen_ids[0, inputs.input_ids.size(1):]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return resp

# ---------------- 仅写日志（status / model_output / f1 / repaired_output） ---------------- #
def _write_step_log(fp, status: str, model_output: str, f1: Optional[float], repaired_output: Optional[str]):
    try:
        rec = {
            "status": status,
            "model_output": model_output,
            "f1": (None if f1 is None else float(f1)),
        }
        if repaired_output is not None:
            rec["repaired_output"] = repaired_output
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
    except Exception:
        pass

# ---------------- BFS 多轮评估（仅用于打分与日志；可展开子节点） ---------------- #
def evaluate_dialog_bfs(
    model, tokenizer, device,
    root_user_json: Dict[str, Any],
    gt_by_turn: List[List[str]],
    max_turns: int = 10,
    max_children_per_node: int = 0,  # 0/None 表示不限制
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    log_fp=None,
):
    from collections import deque

    q = deque()
    root = dict(root_user_json)
    root.setdefault("depth", 0)
    root.setdefault("historical_interactors", [])
    root.setdefault("potential_interactors", [])
    root["gt_by_turn"] = gt_by_turn
    q.append(root)

    statuses: List[str] = []
    f1_list: List[float] = []
    f1_no_inter: List[float] = []
    f1_inter: List[float] = []
    reward_like_all: List[float] = []
    kept_steps = 0

    def build_children(parent_full: Dict[str, Any], model_resp: str) -> List[Dict[str, Any]]:
        preds, status, _ = parse_and_validate(model_resp)
        status = canonical_parse_status(status)
        if status not in ("ok", "ok_repaired") or len(preds) == 0:
            return []
        pot_map = pot_list_to_map(parent_full.get("potential_interactors", []))
        parent_pi = parent_full.get("potential_interactors", [])

        children = []
        for p in preds:
            u = (p.get("user_name") or "").strip()
            # 只校验“在候选池中”
            if (not u) or (u not in pot_map):
                continue
            profile = dict(pot_map[u])
            child = dict(profile)
            child.update(
                {
                    "user_name": u,
                    "content": p.get("content", ""),
                    "depth": int(parent_full.get("depth") or 0) + 1,
                    "historical_interactors": list(parent_full.get("historical_interactors", [])),
                    "gt_by_turn": parent_full.get("gt_by_turn", []),
                    "potential_interactors": list(parent_pi),
                }
            )
            children.append(child)
            if max_children_per_node and len(children) >= max_children_per_node:
                break
        return children

    turns = 0
    while q and turns < max_turns:
        parent = q.popleft()
        depth = int(parent.get("depth") or 0)
        if depth >= len(gt_by_turn):
            turns += 1
            continue

        resp = generate_one_step(
            model=model, tokenizer=tokenizer, device=device,
            user_json=parent, max_new_tokens=max_new_tokens, temperature=temperature
        )

        gold_seq = gt_by_turn[depth] if (isinstance(gt_by_turn, list) and depth < len(gt_by_turn)) else []
        gold_names = list(gold_seq)

        preds, status, repaired_output = parse_and_validate(resp)
        status = canonical_parse_status(status)

        f1_for_log: Optional[float] = None

        if is_parse_fail(status):
            reward_like_all.append(-1.0)
            statuses.append(status)
            kept_steps += 1
            if log_fp is not None:
                _write_step_log(log_fp, status=status, model_output=resp, f1=None, repaired_output=repaired_output)
            turns += 1
            continue

        if status in ("ok", "ok_repaired", "empty"):
            pred_names = [d["user_name"] for d in preds]
        elif status == "no_interaction":
            pred_names = []
        else:
            statuses.append("parse_fail_bad_field")
            reward_like_all.append(-1.0)
            kept_steps += 1
            if log_fp is not None:
                _write_step_log(log_fp, status="parse_fail_bad_field", model_output=resp, f1=None, repaired_output=repaired_output)
            turns += 1
            continue

        f1 = set_f1(pred_names, gold_names)
        f1_for_log = f1
        f1_list.append(f1)
        reward_like_all.append(f1)
        statuses.append(status)
        kept_steps += 1

        if log_fp is not None:
            _write_step_log(log_fp, status=status, model_output=resp, f1=f1_for_log, repaired_output=repaired_output)

        if len(gold_names) == 0:
            f1_no_inter.append(f1)
        else:
            f1_inter.append(f1)

        if status in ("ok", "ok_repaired") and len(preds) > 0:
            children = build_children(parent, resp)
            for c in children:
                q.append(c)

        turns += 1

    return {
        "statuses": statuses,
        "f1_list": f1_list,
        "f1_no_inter": f1_no_inter,
        "f1_inter": f1_inter,
        "reward_like_all": reward_like_all,
        "kept_steps": kept_steps,
    }

# ---------------- 主入口 ---------------- #
def evaluate_parquet_file(
    parquet_path: str,
    model_path: str,
    device: str = "cuda",
    max_samples: int = 100,
    max_turns: int = 10,
    max_children_per_node: int = 0,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    log_path: Optional[str] = "model_outputs.jsonl",
    log_overwrite: bool = False,
):
    print(f"[INFO] load model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype="auto", trust_remote_code=True
    ).eval()

    df = pd.read_parquet(parquet_path)

    # 列名检查
    if ("prompt" not in df.columns) and ("messages" not in df.columns):
        raise ValueError("数据集中缺少 'prompt' 或 'messages' 列（GRPO root-only 需要其一）。")
    if ("reward_model" not in df.columns) and ("reward_model.ground_truth" not in df.columns):
        raise ValueError("数据集中缺少 'reward_model' 或 'reward_model.ground_truth' 列。")

    # 打开日志文件
    fp = None
    if log_path:
        mode = "w" if log_overwrite else "a"
        fp = open(log_path, mode, encoding="utf-8")

    statuses_all: List[str] = []
    f1_all_valid: List[float] = []
    f1_no_inter_all: List[float] = []
    f1_inter_all: List[float] = []
    reward_like_all: List[float] = []
    total_kept = 0

    n_rows = min(int(len(df)), int(max_samples))
    pbar = tqdm(total=n_rows, desc="Evaluating (BFS full-branch)", ncols=100)

    for ridx, row in df.iloc[:n_rows].iterrows():
        try:
            root_user_json = get_root_user_json(row)
            gt_by_turn = get_gt_by_turn(row)

            out = evaluate_dialog_bfs(
                model, tokenizer, device,
                root_user_json=root_user_json,
                gt_by_turn=gt_by_turn,
                max_turns=int(max_turns),
                max_children_per_node=(None if int(max_children_per_node) == 0 else int(max_children_per_node)),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                log_fp=fp,
            )

            statuses_all.extend([canonical_parse_status(s) for s in out["statuses"]])
            f1_all_valid.extend(out["f1_list"])
            f1_no_inter_all.extend(out["f1_no_inter"])
            f1_inter_all.extend(out["f1_inter"])
            reward_like_all.extend(out["reward_like_all"])
            total_kept += out["kept_steps"]

        except Exception as e:
            print(f"[!] row {ridx} failed: {e}")

        pbar.update(1)

    pbar.close()

    if fp is not None:
        try:
            fp.close()
        except Exception:
            pass

    def mean_or_zero(arr: List[float]) -> float:
        return float(np.mean(arr)) if (isinstance(arr, (list, tuple)) and len(arr) > 0) else 0.0

    # 状态计数
    status_counts: Dict[str, int] = {}
    for s in statuses_all:
        status_counts[s] = status_counts.get(s, 0) + 1

    total_steps = len(statuses_all)
    total_parse_fails = sum(cnt for st, cnt in status_counts.items() if is_parse_fail(st))
    parse_fail_rate = (total_parse_fails / total_steps) if total_steps > 0 else 0.0

    ok_n = status_counts.get("ok", 0)
    ok_rep_n = status_counts.get("ok_repaired", 0)
    empty_n = status_counts.get("empty", 0)
    noint_n = status_counts.get("no_interaction", 0)

    print("\n=================== Evaluation Summary (BFS 全分支) ===================")
    print(f"✅ 评估对话数: {n_rows}")
    print(f"✅ 采样到的有效 step 数(kept_steps): {total_kept}")
    print(f"• 状态分布(总步数={total_steps}): ok={ok_n} ok_repaired={ok_rep_n} empty={empty_n} "
          f"no_interaction={noint_n} parse_fail_total={total_parse_fails} "
          f"(parse_fail_rate={parse_fail_rate:.4f})")

    if total_parse_fails > 0:
        print("\n🔍 解析失败细分：")
        for st, cnt in sorted(status_counts.items(), key=lambda x: (-x[1], x[0])):
            if is_parse_fail(st):
                rate = cnt / total_steps if total_steps > 0 else 0.0
                print(f"  - {st}: {cnt} ({rate:.4%})")

    print(f"\n📊 仅可解析样本（排除所有 parse_fail）的 F1：")
    print(f"  - F1(无交互金标)   = {mean_or_zero(f1_no_inter_all):.4f}  (n={len(f1_no_inter_all)})")
    print(f"  - F1(有交互金标)   = {mean_or_zero(f1_inter_all):.4f}      (n={len(f1_inter_all)})")
    print(f"  - F1(全部可解析)   = {mean_or_zero(f1_all_valid):.4f}      (n={len(f1_all_valid)})")

    print(f"\n🧪 训练口径（含 parse_fail=-1）：{mean_or_zero(reward_like_all):.4f}  (n={len(reward_like_all)})")
    if log_path:
        print(f"📝 已将逐步输出写入: {log_path}")
    print("=======================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="输入 GRPO root-only Parquet 文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型路径（如 checkpoints/.../global_step_xxx）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")
    parser.add_argument("--max_samples", type=int, default=100, help="最大样本条数（按行计）")
    parser.add_argument("--max_turns", type=int, default=10, help="每个对话的最大展开步数（BFS 层数之和的上限）")
    parser.add_argument("--max_children_per_node", type=int, default=0, help="每个节点最多展开的子节点数；0 表示不限制")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="生成上限 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help=">0 启用采样")
    parser.add_argument("--log_path", type=str, default="model_outputs.jsonl", help="逐步输出 JSONL 文件路径")
    parser.add_argument("--log_overwrite", action="store_true", help="若指定，则覆盖已有文件；默认追加写入")

    args = parser.parse_args()

    evaluate_parquet_file(
        parquet_path=args.data,
        model_path=args.model,
        device=args.device,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        max_children_per_node=args.max_children_per_node,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_path=args.log_path,
        log_overwrite=args.log_overwrite,
    )
