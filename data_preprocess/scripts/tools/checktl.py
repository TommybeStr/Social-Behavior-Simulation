#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计数据构造输出的 Parquet 中，user 段结尾为 <TL0> / <TL1> 的样本数量。
- 默认使用 node_depth 列（0/1）做快速统计；
- 可选 --verify：逐行解析 messages['user'].content 的结尾是否为 <TL0>/<TL1>，并与 node_depth 交叉校验。

依赖：pyarrow
pip install pyarrow
"""

import os
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

import pyarrow.parquet as pq

TL0 = "<TL0>"
TL1 = "<TL1>"

def find_parquet_paths(args) -> List[str]:
    paths = []
    if args.dir:
        for name in os.listdir(args.dir):
            if name.endswith(".parquet"):
                paths.append(os.path.join(args.dir, name))
    if args.paths:
        paths.extend(args.paths)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit("未找到任何 .parquet 文件。请用 --dir 或 --paths 指定。")
    paths.sort()
    return paths

def scan_file(path: str, verify: bool=False, verify_limit: int=None) -> Dict[str, float]:
    """
    返回指标字典：
      - rows: 行数
      - depth0, depth1: 基于 node_depth 的计数
      - v_tl0, v_tl1, v_unknown: 校验模式下基于 user 文本结尾的计数
      - mismatch: （仅 verify）node_depth 与文本结尾不一致的行数
    """
    pf = pq.ParquetFile(path)
    need_cols = ["node_depth"]
    if verify:
        need_cols.append("messages")
    for c in need_cols:
        if c not in pf.schema.names:
            raise RuntimeError(f"{path} 缺少列: {c}")

    rows = 0
    depth0 = 0
    depth1 = 0

    v_tl0 = 0
    v_tl1 = 0
    v_unknown = 0
    mismatch = 0

    processed_for_verify = 0

    for rg in range(pf.num_row_groups):
        cols = ["node_depth"] + (["messages"] if verify else [])
        table = pf.read_row_group(rg, columns=cols)

        depths = table.column("node_depth").to_pylist()  # List[int]
        rows += len(depths)
        depth0 += sum(1 for d in depths if int(d) == 0)
        depth1 += sum(1 for d in depths if int(d) == 1)

        if verify:
            msgs = table.column("messages").to_pylist()  # List[List[{'role','content','loss'}]]
            for d, mlist in zip(depths, msgs):
                if verify_limit is not None and processed_for_verify >= verify_limit:
                    break
                processed_for_verify += 1

                # 找到 role=user 的那条
                user_content = None
                if isinstance(mlist, list):
                    for m in mlist:
                        try:
                            if (m.get("role") == "user") and ("content" in m):
                                user_content = m["content"]
                                break
                        except Exception:
                            pass

                tail = None
                if isinstance(user_content, str):
                    s = user_content.rstrip()
                    if s.endswith(TL1):
                        v_tl1 += 1
                        tail = 1
                    elif s.endswith(TL0):
                        v_tl0 += 1
                        tail = 0
                    else:
                        v_unknown += 1
                        tail = None
                else:
                    v_unknown += 1
                    tail = None

                # 与 node_depth 交叉校验
                if tail is not None and int(d) in (0, 1) and tail != int(d):
                    mismatch += 1
            if verify_limit is not None and processed_for_verify >= verify_limit:
                break

    return {
        "rows": rows,
        "depth0": depth0,
        "depth1": depth1,
        "v_tl0": v_tl0,
        "v_tl1": v_tl1,
        "v_unknown": v_unknown,
        "mismatch": mismatch,
        "v_checked": processed_for_verify if verify else 0,
    }

def pct(n: float, d: float) -> str:
    return "0.00%" if d <= 0 else f"{(100.0 * n / d):.2f}%"

def main():
    ap = argparse.ArgumentParser(description="统计 TL0 / TL1 分布（基于 node_depth 或校验 user 文本结尾）")
    ap.add_argument("--dir", type=str, help="包含 parquet 的目录")
    ap.add_argument("--paths", nargs="*", help="指定若干 parquet 文件路径")
    ap.add_argument("--verify", action="store_true", help="逐行解析 messages 的 user 结尾是否为 <TL0>/<TL1> 并交叉校验")
    ap.add_argument("--verify-limit", type=int, default=None, help="校验模式下最多检查多少行（加快调试）")
    args = ap.parse_args()

    files = find_parquet_paths(args)
    print("将统计以下文件：")
    for p in files:
        print(" -", p)

    total = defaultdict(float)

    print("\n================ 逐文件统计 ================\n")
    for p in files:
        stats = scan_file(p, verify=args.verify, verify_limit=args.verify_limit)
        total_rows = stats["rows"]
        d0, d1 = stats["depth0"], stats["depth1"]
        print(f"[{os.path.basename(p)}]")
        print(f"rows={total_rows} | node_depth: TL0={d0} ({pct(d0,total_rows)}), TL1={d1} ({pct(d1,total_rows)})")
        if args.verify:
            v_checked = stats["v_checked"]
            v0, v1, vun = stats["v_tl0"], stats["v_tl1"], stats["v_unknown"]
            mm = stats["mismatch"]
            print(f"verify_checked={v_checked} | tail: TL0={v0} ({pct(v0,v_checked)}), TL1={v1} ({pct(v1,v_checked)}), UNKNOWN={vun} ({pct(vun,v_checked)}) | mismatch={mm}")
        print()

        for k, v in stats.items():
            total[k] += v

    print("================ 汇总统计 ================\n")
    tr = total["rows"]
    td0, td1 = total["depth0"], total["depth1"]
    print(f"[ALL] rows={int(tr)}")
    print(f"node_depth: TL0={int(td0)} ({pct(td0,tr)}), TL1={int(td1)} ({pct(td1,tr)})")

    if args.verify:
        v_checked = int(total["v_checked"])
        v0, v1, vun = int(total["v_tl0"]), int(total["v_tl1"]), int(total["v_unknown"])
        mm = int(total["mismatch"])
        print(f"verify_checked={v_checked} | tail: TL0={v0} ({pct(v0,v_checked)}), TL1={v1} ({pct(v1,v_checked)}), UNKNOWN={vun} ({pct(vun,v_checked)})")
        print(f"node_depth vs tail mismatch: {mm}")

if __name__ == "__main__":
    main()
