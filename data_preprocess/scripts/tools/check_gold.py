#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 json/jsonl GRPO 数据集中：
1) 非一层（depth>=2）gold 的样本数量与分布（保留原有口径）
2) 每层 gold 用户数（去重后）的统计：
   - 按出现次数累计：对每条样本的每层先去重，再跨样本累加
   - 全局唯一：跨样本、跨层的全局去重
3) 第二层（业务口径 depth=2）gold 用户数在“总用户数”中的占比：
   - 出现次数累计口径
   - 全局唯一口径
"""

import json
import argparse
from typing import Any, Dict, List, Iterable, Tuple
from collections import defaultdict
import sys

NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"

def json_try_load(s: str):
    s = (s or "").strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def dedup_keep_order(seq: Iterable[Any]):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def to_names(x) -> List[str]:
    """把宽松结构（str / list[dict or str] / dict）统一成用户名列表（去空去重保序）"""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s == "[]" or s == NO_INTERACTION_STR:
            return []
        parsed = json_try_load(s)
        if isinstance(parsed, list):
            return to_names(parsed)
        return [s]
    out: List[str] = []
    if isinstance(x, dict):
        n = (x.get("user_name") or "").strip()
        return [n] if n else []
    if isinstance(x, list):
        for it in x:
            if isinstance(it, str):
                s = it.strip()
                if not s or s == "[]" or s == NO_INTERACTION_STR:
                    continue
                parsed = json_try_load(s)
                if isinstance(parsed, list):
                    out.extend(to_names(parsed))
                else:
                    out.append(s)
            elif isinstance(it, dict):
                n = (it.get("user_name") or "").strip()
                if n:
                    out.append(n)
    return [n for n in dedup_keep_order(out) if n]

def _layer_is_list_of_parent_maps(level: Any) -> bool:
    """检测 level 是否形如 list[{parent: [children]}, ...]"""
    if not isinstance(level, list):
        return False
    for it in level:
        if not isinstance(it, dict) or len(it) != 1:
            return False
        (k, v), = it.items()
        if not isinstance(k, str) or not isinstance(v, list):
            return False
    return True

def extract_children_names(level: Any) -> List[str]:
    """
    提取某一层 level 的 gold 子节点用户名（去重保序）。
    兼容三种结构：list-of-maps / dict / list。
    """
    names: List[str] = []
    if _layer_is_list_of_parent_maps(level):
        for m in level:
            (_, ch), = m.items()
            names.extend(to_names(ch))
    elif isinstance(level, dict):
        for _, ch in level.items():
            names.extend(to_names(ch))
    elif isinstance(level, list):
        names.extend(to_names(level))
    else:
        return []
    return [n for n in dedup_keep_order(names) if n]

def count_children_in_level(level: Any) -> int:
    """保留兼容接口：返回该层去重后的人数"""
    return len(extract_children_names(level))

def extract_cond_gt_by_turn(record: Dict[str, Any]) -> List[Any]:
    """
    从 record 中提取 cond_gt_by_turn：
    - 先找 data['prompt'][first user]['content'] 里的 JSON 再取 reward_model.ground_truth.cond_gt_by_turn
    - 或兼容 data['messages'] 结构
    """
    prompt = record.get("prompt")
    messages = record.get("messages")
    dialog = None
    if isinstance(prompt, list):
        dialog = prompt
    elif isinstance(messages, list):
        dialog = messages

    if not dialog:
        return []

    user_json_str = None
    for m in dialog:
        if isinstance(m, dict) and m.get("role") == "user":
            user_json_str = m.get("content")
            break
    if not user_json_str or not isinstance(user_json_str, str):
        return []

    root_user = json_try_load(user_json_str)
    if not isinstance(root_user, dict):
        return []

    rm = (root_user.get("reward_model") or {})
    gt = (rm.get("ground_truth") or {})
    cond = gt.get("cond_gt_by_turn")
    if isinstance(cond, list):
        return cond
    return []

def read_records(path: str) -> Iterable[Dict[str, Any]]:
    """读取 JSON 数组或 JSONL 文件，逐条产出 dict 记录。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        if not head:
            return
    with open(path, "r", encoding="utf-8") as f:
        if head == "[":
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj

def record_id_fallback(rec: Dict[str, Any], idx: int) -> str:
    info = rec.get("sft_chunk_info") or {}
    if isinstance(info, dict):
        return info.get("sample_id") or info.get("record_id") or f"row{idx}"
    return f"row{idx}"

def main():
    parser = argparse.ArgumentParser(description="统计 GRPO json/jsonl 数据集中非一层（depth>=2）gold 及各层 gold 用户数与占比")
    parser.add_argument("--data", required=True, help="输入文件路径（.json 或 .jsonl）")
    parser.add_argument("--show_examples", type=int, default=10, help="最多展示多少条含非一层 gold 的样本 id（默认10）")
    args = parser.parse_args()

    total = 0
    records_with_nonfirst_gold = 0
    per_depth_nonempty = defaultdict(int)  # 业务口径层号: 该层 gold 非空的样本数
    example_ids: List[str] = []

    # 新增统计 —— 每层用户数
    sum_gold_by_depth = defaultdict(int)       # 按出现次数累计（样本内去重后再累加）
    unique_names_by_depth = defaultdict(set)   # 按层的全局唯一
    global_unique_all = set()                  # 跨层、跨样本的全局唯一
    sum_gold_all_depths = 0                    # 按出现次数累计的总和

    for idx, rec in enumerate(read_records(args.data)):
        total += 1
        cond = extract_cond_gt_by_turn(rec)
        if not isinstance(cond, list) or len(cond) == 0:
            continue

        has_nonfirst = False

        # 遍历所有层（内部0基），业务口径层号= d+1
        for d in range(len(cond)):
            level = cond[d]
            names = extract_children_names(level)  # 当前样本该层去重后的名单
            depth_label = d + 1

            # —— 层内统计（样本内去重）
            layer_count = len(names)
            if layer_count > 0:
                sum_gold_by_depth[depth_label] += layer_count
                sum_gold_all_depths += layer_count

                # —— 全局唯一
                unique_names_by_depth[depth_label].update(names)
                global_unique_all.update(names)

            # —— 原有“非一层（depth>=2）非空样本计数”
            if d >= 1 and layer_count > 0:
                has_nonfirst = True
                per_depth_nonempty[depth_label] += 1

        if has_nonfirst:
            records_with_nonfirst_gold += 1
            if len(example_ids) < args.show_examples:
                example_ids.append(record_id_fallback(rec, idx))

    # ===== 输出 =====
    print("\n=== 非一层（depth>=2）gold 统计（原口径） ===")
    print(f"总样本数: {total}")
    print(f"含非一层 gold 的样本数: {records_with_nonfirst_gold}")
    ratio = (records_with_nonfirst_gold / total) if total > 0 else 0.0
    print(f"占比: {ratio:.2%}")

    if per_depth_nonempty:
        print("\n按业务口径层号统计（仅列出 depth>=2 且该层 gold 非空的样本计数）：")
        for depth_label in sorted(per_depth_nonempty.keys()):
            if depth_label >= 2:
                print(f"  depth={depth_label}: {per_depth_nonempty[depth_label]}")

    # —— 新增：各层 gold 用户数（两种口径）
    print("\n=== 各层 gold 用户数统计 ===")
    if sum_gold_by_depth:
        print("按出现次数累计（样本内去重后跨样本累加）：")
        for depth_label in sorted(sum_gold_by_depth.keys()):
            print(f"  depth={depth_label}: {sum_gold_by_depth[depth_label]}")
        print(f"合计（全部层，出现次数累计）: {sum_gold_all_depths}")

    if unique_names_by_depth:
        print("\n全局唯一（跨样本、跨层全局去重的分层人数）：")
        for depth_label in sorted(unique_names_by_depth.keys()):
            print(f"  depth={depth_label}: {len(unique_names_by_depth[depth_label])}")
        print(f"合计（全部层，全局唯一）: {len(global_unique_all)}")

    # —— 新增：第二层占比
    depth2_sum = sum_gold_by_depth.get(2, 0)
    depth2_unique = len(unique_names_by_depth.get(2, set()))
    occ_ratio = (depth2_sum / sum_gold_all_depths) if sum_gold_all_depths > 0 else 0.0
    uniq_ratio = (depth2_unique / len(global_unique_all)) if len(global_unique_all) > 0 else 0.0

    print("\n=== 第二层（depth=2）在总 gold 用户数中的占比 ===")
    print(f"- 按出现次数累计：第二层={depth2_sum} / 总计={sum_gold_all_depths}  占比={occ_ratio:.2%}")
    print(f"- 全局唯一：第二层唯一={depth2_unique} / 总唯一={len(global_unique_all)}  占比={uniq_ratio:.2%}")

    if example_ids:
        print("\n示例样本（最多显示 {} 条）：".format(len(example_ids)))
        for rid in example_ids:
            print("  -", rid)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
