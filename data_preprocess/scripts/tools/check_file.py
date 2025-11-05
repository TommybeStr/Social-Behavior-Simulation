#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按层统计 SFT 构造数据集（带 depth 的 Parquet）中的 gold 交互：
- 有效交互数（每个 gold 用户计 1）
- 无效交互数（gold 为 [] 计 1）
- 有效交互中的类型分布：评论数、转发数

用法：
  python stats_gold_by_depth.py --inputs train_depth.parquet val_depth.parquet test_depth.parquet \
      --out_csv stats_by_depth.csv

也支持通配符：
  python stats_gold_by_depth.py --inputs "data/*depth*.parquet"
"""

import argparse
import json
import sys
import glob
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

# 有些数据源里可能出现 "转发" 或 "转发微博"；统一归并为 "转发"
RT_ALIASES = {"转发", "转发微博"}

def _norm_type(t: str) -> str:
    t = (t or "").strip()
    return "转发" if t in RT_ALIASES else "评论"

def _iter_assistant_messages_with_depth(messages_col):
    """
    从一行的 messages（list<struct{role,content,loss,depth}>）里提取所有 (depth, content)
    其中 role == 'assistant'。
    """
    # messages_col 是 Python 列表，每个元素是 dict：{'role':..., 'content':..., 'loss':..., 'depth': ...}
    for m in messages_col or []:
        try:
            if (m.get("role") == "assistant"):
                yield m.get("depth"), m.get("content")
        except Exception:
            continue

def _safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="一个或多个 Parquet 路径；支持 shell 通配（记得加引号防止被 shell 提前展开失败）")
    ap.add_argument("--out_csv", default="", help="可选：把分层统计结果另存为 CSV")
    args = ap.parse_args()

    # 展开通配
    paths = []
    for p in args.inputs:
        paths.extend(glob.glob(p))
    paths = list(dict.fromkeys(paths))  # 去重保序

    if not paths:
        print("[ERR] 未找到任何输入文件。", file=sys.stderr)
        sys.exit(1)

    # 结果累加容器（按 depth）
    by_depth = {
        # depth: dict
        #   'valid_interactions': int
        #   'invalid_interactions': int
        #   'comment_in_valid': int
        #   'retweet_in_valid': int
    }
    # 用 defaultdict 便于累加
    valid_interactions = defaultdict(int)
    invalid_interactions = defaultdict(int)
    comment_in_valid = defaultdict(int)
    retweet_in_valid = defaultdict(int)

    total_rows = 0
    rows_without_depth = 0

    for path in paths:
        try:
            table: pa.Table = pq.read_table(path)
        except Exception as e:
            print(f"[WARN] 无法读取 {path}: {e}", file=sys.stderr)
            continue

        if "messages" not in table.column_names:
            print(f"[WARN] 文件 {path} 缺少 'messages' 列，跳过。", file=sys.stderr)
            continue

        # 转为 Python 迭代（arrow 大表可分批，但这里通常可一次性取出）
        messages_arr = table["messages"].to_pylist()
        total_rows += len(messages_arr)

        for msgs in messages_arr:
            for depth, content in _iter_assistant_messages_with_depth(msgs):
                # 有些 no-depth 版本会移除 depth，这里跳过
                if depth is None:
                    rows_without_depth += 1
                    continue

                arr = _safe_json_load(content)
                # 遇到非 JSON 数组（数据异常），跳过该条 assistant 消息
                if not isinstance(arr, list):
                    continue

                if len(arr) == 0:
                    # 无效交互：gold 为 []
                    invalid_interactions[depth] += 1
                else:
                    # 有效交互：每个 gold 用户记一次
                    valid_interactions[depth] += len(arr)

                    # 类型分布只统计有效交互内的元素
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        typ = _norm_type(item.get("type", "评论"))
                        if typ == "评论":
                            comment_in_valid[depth] += 1
                        else:
                            retweet_in_valid[depth] += 1

    # 汇总并输出
    depths = sorted(set(list(valid_interactions.keys()) + list(invalid_interactions.keys())
                        + list(comment_in_valid.keys()) + list(retweet_in_valid.keys())))

    if not depths:
        print("[INFO] 没有可统计的带 depth 的 assistant 消息。请确认输入为“带 depth”的 Parquet。")
        if rows_without_depth > 0:
            print(f"[HINT] 检测到 {rows_without_depth} 条消息缺少 depth，可能是你传入了 no-depth 版本。")
        sys.exit(0)

    print("=== 分层统计（基于 gold/assistant 输出）===")
    print("depth, 有效交互数(sum of gold users), 无效交互数(#gold==[]), 其中有效交互-评论数, 其中有效交互-转发数")
    for d in depths:
        v = valid_interactions[d]
        iv = invalid_interactions[d]
        c = comment_in_valid[d]
        r = retweet_in_valid[d]
        print(f"{d}, {v}, {iv}, {c}, {r}")

    # 可选导出 CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["depth", "valid_interactions", "invalid_interactions", "comment_in_valid", "retweet_in_valid"])
            for d in depths:
                w.writerow([d, valid_interactions[d], invalid_interactions[d], comment_in_valid[d], retweet_in_valid[d]])
        print(f"[OK] 已写出 CSV: {args.out_csv}")

    # 汇总提示
    if rows_without_depth > 0:
        print(f"[NOTE] 有 {rows_without_depth} 条 assistant 消息缺少 depth（可能是 no-depth 数据），已跳过。")

if __name__ == "__main__":
    main()
