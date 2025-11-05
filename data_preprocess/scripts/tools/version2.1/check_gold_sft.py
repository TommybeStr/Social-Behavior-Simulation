#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT Parquet（messages 列）gold 检测脚本 —— 适配你的数据构造格式

功能：
1) 读取一个或多个（带 depth 的）Parquet 文件，解析 messages 中 assistant 的 JSON 数组作为 gold。
2) 统计：
   - 各层 gold 用户数（出现次数累计 vs 全局唯一）
   - 第二层（depth=2）在总 gold 用户数中的占比（两种口径）
   - 含非一层（depth>=2）gold 的样本数与占比（以 row id 为单位）
   - 按层（depth>=2）gold 非空的样本计数
   - 检测单条 assistant 输出内的重复 user_name（如有）
3) 对去 depth 版本也可运行，但会跳过所有与 depth 相关的统计。

用法：
  python sft_gold_check.py \
      --parquet train_depth.parquet val_depth.parquet test_depth.parquet \
      --show_examples 10
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

import pyarrow.parquet as pq
import pyarrow as pa

# tqdm（若未安装则降级为空操作）
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"


# ----------------- 小工具 -----------------
def json_try_load(s: str):
    s = (s or "").strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def dedup_keep_order(seq: Iterable[Any]) -> List[Any]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_assistant_user_names(assistant_content: Any) -> List[str]:
    """
    解析 assistant JSON 数组，提取 user_name 列表（去空、去重、保序）。
    允许混合：字符串 / dict / list。
    """
    if assistant_content is None:
        return []
    # 如果是字符串，尝试解析
    if isinstance(assistant_content, str):
        s = assistant_content.strip()
        if not s or s == "[]" or s == NO_INTERACTION_STR:
            return []
        parsed = json_try_load(s)
        if parsed is None:
            # 不是 JSON，就当成一个用户名
            return [s]
        assistant_content = parsed

    names: List[str] = []
    if isinstance(assistant_content, list):
        for it in assistant_content:
            if isinstance(it, dict):
                n = (it.get("user_name") or "").strip()
                if n:
                    names.append(n)
            elif isinstance(it, str):
                ss = it.strip()
                if not ss or ss == "[]" or ss == NO_INTERACTION_STR:
                    continue
                parsed = json_try_load(ss)
                if isinstance(parsed, list):
                    names.extend(extract_assistant_user_names(parsed))
                else:
                    names.append(ss)
    elif isinstance(assistant_content, dict):
        # 容错：若出现 dict 直接带 user_name
        n = (assistant_content.get("user_name") or "").strip()
        if n:
            names.append(n)

    return [n for n in dedup_keep_order(names) if n]


def has_depth_field(messages_item: Dict[str, Any]) -> bool:
    # 判断该 Parquet 是否包含 depth（检查任意一条 message 是否带 depth 字段）
    return "depth" in messages_item if isinstance(messages_item, dict) else False


# ----------------- 主逻辑 -----------------
def process_parquet(files: List[str], show_examples: int):
    # 统计量
    total_rows = 0                          # Parquet 行数（样本条数）
    rows_with_nonfirst_gold = 0             # 有 depth>=2 的 gold 的样本数（按 id）

    per_depth_nonempty_rows = defaultdict(int)   # 某层 gold 非空的样本计数（depth>=2）
    sum_gold_by_depth = defaultdict(int)         # 各层 gold 人数（出现次数累计；每 message 先去重）
    unique_names_by_depth: Dict[int, Set[str]] = defaultdict(set)  # 各层全局唯一
    global_unique_all: Set[str] = set()         # 跨层全局唯一
    sum_gold_all_depths = 0                      # 按出现次数累计的总 gold 量

    example_ids: List[str] = []                 # 抽样展示含非一层 gold 的样本 id
    duplicate_hits = 0                          # 单条 assistant 内部重复情况出现的次数（排查用）
    has_depth_any = None                        # 该批文件是否带 depth 字段（None=未知，True/False）

    # 读取每个 Parquet
    for path in files:
        pf = pq.ParquetFile(path)
        # 逐 row group 读取，节省内存
        for rg in range(pf.num_row_groups):
            table: pa.Table = pf.read_row_group(rg, columns=["id", "messages"])
            ids = table.column("id").to_pylist()
            msgs_col = table.column("messages").to_pylist()

            for row_idx, (sid, msg_list) in enumerate(zip(ids, msgs_col)):
                total_rows += 1
                # msg_list 是 list[struct] -> 转为 python dict
                # pyarrow 会给 dict-like 的 Python 对象
                messages: List[Dict[str, Any]] = []
                for m in msg_list or []:
                    # m 可能是 dict-like；转标准 dict
                    messages.append({k: m.get(k) for k in m.keys()})

                # 第一次确定是否带 depth
                if has_depth_any is None:
                    has_depth_any = any(has_depth_field(m) for m in messages)

                row_has_nonfirst = False
                # 为了一个样本对“某层非空样本计数”只+1次，需要记录本行出现过哪些层
                row_nonempty_depths: Set[int] = set()

                # 遍历 assistant 消息
                for m in messages:
                    if (m.get("role") != "assistant"):
                        continue

                    content = m.get("content")
                    names = extract_assistant_user_names(content)   # 先去重
                    if not names:
                        continue

                    # 检查单条 assistant 内部重复（若你构造脚本保证不重复，这里应为 0）
                    if len(names) != len(set(names)):
                        duplicate_hits += 1

                    if has_depth_any:
                        d = m.get("depth")
                        try:
                            depth_label = int(d) if d is not None else None
                        except Exception:
                            depth_label = None
                    else:
                        depth_label = None

                    # —— 与 depth 无关的总量：累加
                    sum_gold_all_depths += len(names)
                    global_unique_all.update(names)

                    # —— 有 depth 的情况下，做分层统计
                    if depth_label is not None:
                        sum_gold_by_depth[depth_label] += len(names)
                        unique_names_by_depth[depth_label].update(names)

                        # 非一层统计
                        if depth_label >= 1 and len(names) > 0:
                            row_has_nonfirst = True
                            row_nonempty_depths.add(depth_label)

                # 行级收尾
                if row_has_nonfirst:
                    rows_with_nonfirst_gold += 1
                    if len(example_ids) < show_examples:
                        example_ids.append(str(sid))

                for d in row_nonempty_depths:
                    per_depth_nonempty_rows[d] += 1

    # ========== 输出 ==========
    print("\n=== 基本信息 ===")
    print(f"读取 Parquet 文件数: {len(files)}")
    print(f"总样本（行）数: {total_rows}")
    print(f"是否检测到 depth 字段: {has_depth_any}")

    # 非一层（depth>=2）
    print("\n=== 非一层（depth>=2）gold 统计（按样本 id） ===")
    print(f"含非一层 gold 的样本数: {rows_with_nonfirst_gold}")
    ratio = (rows_with_nonfirst_gold / total_rows) if total_rows > 0 else 0.0
    print(f"占比: {ratio:.2%}")

    if has_depth_any:
        print("\n按层（depth>=2）gold 非空的样本计数：")
        for depth_label in sorted(per_depth_nonempty_rows.keys()):
            if depth_label >= 2:
                print(f"  depth={depth_label}: {per_depth_nonempty_rows[depth_label]}")

    # 各层 gold 用户数（两种口径）
    print("\n=== 各层 gold 用户数统计 ===")
    if has_depth_any and sum_gold_by_depth:
        print("按出现次数累计（单条 assistant 内先去重，再跨样本累加）：")
        for depth_label in sorted(sum_gold_by_depth.keys()):
            print(f"  depth={depth_label}: {sum_gold_by_depth[depth_label]}")
        print(f"合计（全部层，出现次数累计）: {sum_gold_all_depths}")
    else:
        # 即使没有 depth，也给出整体合计
        print(f"合计（不分层，出现次数累计）: {sum_gold_all_depths}")

    if has_depth_any and unique_names_by_depth:
        print("\n全局唯一（跨样本、跨层全局去重的分层人数）：")
        for depth_label in sorted(unique_names_by_depth.keys()):
            print(f"  depth={depth_label}: {len(unique_names_by_depth[depth_label])}")
        print(f"合计（全部层，全局唯一）: {len(global_unique_all)}")
    else:
        print(f"\n合计（不分层，全局唯一）: {len(global_unique_all)}")

    # 第二层占比
    if has_depth_any:
        depth2_sum = sum_gold_by_depth.get(2, 0)
        depth2_unique = len(unique_names_by_depth.get(2, set()))
        occ_ratio = (depth2_sum / sum_gold_all_depths) if sum_gold_all_depths > 0 else 0.0
        uniq_ratio = (depth2_unique / len(global_unique_all)) if len(global_unique_all) > 0 else 0.0

        print("\n=== 第二层（depth=2）在总 gold 用户数中的占比 ===")
        print(f"- 按出现次数累计：第二层={depth2_sum} / 总计={sum_gold_all_depths}  占比={occ_ratio:.2%}")
        print(f"- 全局唯一：第二层唯一={depth2_unique} / 总唯一={len(global_unique_all)}  占比={uniq_ratio:.2%}")
    else:
        print("\n未检测到 depth 字段，跳过“第二层占比”与分层细节。")

    # 抽样样本
    if example_ids:
        print("\n示例样本（含非一层 gold，最多显示 {} 条）：".format(len(example_ids)))
        for rid in example_ids:
            print("  -", rid)

    # 重复检测
    if duplicate_hits > 0:
        print(f"\n[警告] 检测到 {duplicate_hits} 条 assistant 输出内部存在重复 user_name（你脚本原则上不应有内部重复）。")
    else:
        print("\n[检查] 未发现 assistant 输出内部的重复 user_name。")


def main():
    parser = argparse.ArgumentParser(description="SFT Parquet gold 检测（适配你的数据构造脚本输出）")
    parser.add_argument("--parquet", nargs="+", required=True, help="一个或多个 Parquet 文件（建议使用带 depth 版本）")
    parser.add_argument("--show_examples", type=int, default=10, help="最多展示多少条含非一层 gold 的样本 id（默认10）")
    args = parser.parse_args()
    process_parquet(args.parquet, args.show_examples)


if __name__ == "__main__":
    main()
