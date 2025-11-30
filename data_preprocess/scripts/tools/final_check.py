#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 depth 统计 SFT 数据中 assistant items 的 type 分布：

- 对每条样本的 messages：
  - 找到 user 块，解析该 user 所在的 depth；
  - 读取紧跟其后的 assistant 块（JSON 数组），其中每个 item 有字段 "type": 0/1/2；
  - 按 depth 统计所有 item 的 type 计数。

输出：
- 每个 depth 上 type=0 / 1 / 2 的数量（item 数量）。
- 可选导出 CSV。
"""

import os
import re
import json
import glob
import argparse
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd

# ====== 常量 / 正则 ======
_PSEP_TOKEN = "<|psep|>"
_PSEP_BLOCK_RE = re.compile(r"<POTENTIAL_SPANS>\s*(.*?)\s*</POTENTIAL_SPANS>", re.DOTALL)
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def safe_json_loads(s: Any):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

# ====== IO ======
def collect_paths(inputs: List[str]) -> List[str]:
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(glob.glob(os.path.join(inp, "*.parquet")))
        else:
            paths.extend(glob.glob(inp))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def iter_parquet(paths: List[str]) -> Iterable[Tuple[str, pd.DataFrame]]:
    for p in paths:
        df = pd.read_parquet(p, engine="pyarrow")
        # 仅保留可能用到的列；若缺失也不强求
        keep = [c for c in ["id", "messages"] if c in df.columns]
        if not keep:
            keep = list(df.columns)
        yield p, df[keep].copy()

def to_list(obj: Any) -> List[Any]:
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            pass
    try:
        import pyarrow as pa
        if isinstance(obj, (pa.Array, pa.ChunkedArray, pa.ListScalar)):
            return obj.to_pylist()
    except Exception:
        pass
    j = safe_json_loads(obj)
    return j if isinstance(j, list) else []

# ====== 解析 user / assistant ======
def parse_depth_from_user_plain(s: Any) -> int:
    """
    从 user.content 里的 <POTENTIAL_SPANS> 中解析 depth：
    - 取 block 里第一个 JSON 的 "depth" 字段。
    """
    if not isinstance(s, str):
        return 0
    m = _PSEP_BLOCK_RE.search(s)
    if not m:
        return 0
    body = m.group(1)
    parts = body.split(_PSEP_TOKEN)
    for part in parts:
        part = part.strip()
        if not part or not part.startswith("{"):
            continue
        obj = safe_json_loads(part)
        if isinstance(obj, dict) and "depth" in obj:
            try:
                return int(obj.get("depth", 0))
            except Exception:
                return 0
    return 0

def parse_assistant_items(assistant_content: Any) -> List[Dict[str, Any]]:
    """
    将 assistant.content 解析为 item list：
    - 标准情况：JSON 数组字符串；
    - 兼容已是 list/dict 的情况。
    """
    if isinstance(assistant_content, list):
        return [it for it in assistant_content if isinstance(it, dict)]
    if isinstance(assistant_content, dict):
        # 兼容一些可能的封装
        for key in ("items", "output", "candidates", "data", "list", "arr"):
            if key in assistant_content and isinstance(assistant_content[key], list):
                return [it for it in assistant_content[key] if isinstance(it, dict)]
        if any(k in assistant_content for k in ("type", "user_name", "username", "name")):
            return [assistant_content]
        return []
    arr = safe_json_loads(assistant_content)
    if isinstance(arr, list):
        return [it for it in arr if isinstance(it, dict)]
    if isinstance(arr, dict):
        return parse_assistant_items(arr)
    return []

# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser(description="按 depth 统计 SFT 数据中 type=0/1/2 分布")
    ap.add_argument("--inputs", nargs="+", required=True, help="Parquet 路径/目录/通配符，可多个")
    ap.add_argument("--min_depth", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=-1, help="若 <0 则不限制最大 depth")
    ap.add_argument("--csv_out", type=str, default="", help="可选：导出 CSV 路径")
    args = ap.parse_args()

    paths = collect_paths(args.inputs)
    if not paths:
        print("未找到任何 Parquet；请检查 --inputs。")
        return

    # depth -> Counter({0: x, 1: y, 2: z, "other": k})
    depth_type_counter: Dict[int, Counter] = defaultdict(Counter)

    for p, df in iter_parquet(paths):
        print(f"[INFO] 处理文件: {p}  rows={len(df)}")

        for _, row in df.iterrows():
            msgs = to_list(row.get("messages"))
            if not msgs:
                continue

            i = 0
            while i < len(msgs):
                m = msgs[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    user_plain = m.get("content", "")
                    user_depth = parse_depth_from_user_plain(user_plain)

                    # depth 过滤
                    if user_depth < args.min_depth or (args.max_depth >= 0 and user_depth > args.max_depth):
                        # 跳过其相邻 assistant（如果有）
                        if i + 1 < len(msgs) and isinstance(msgs[i + 1], dict) and msgs[i + 1].get("role") == "assistant":
                            i += 2
                        else:
                            i += 1
                        continue

                    # 相邻 assistant
                    items: List[Dict[str, Any]] = []
                    if i + 1 < len(msgs) and isinstance(msgs[i + 1], dict) and msgs[i + 1].get("role") == "assistant":
                        items = parse_assistant_items(msgs[i + 1].get("content", ""))

                    # 统计该 depth 上所有 items 的 type
                    for it in items:
                        try:
                            t = int(it.get("type", 0))
                        except Exception:
                            t = -1
                        if t in (0, 1, 2):
                            depth_type_counter[user_depth][t] += 1
                        else:
                            depth_type_counter[user_depth]["other"] += 1

                    # 跳过 assistant
                    i += 2
                else:
                    i += 1

    # ====== 输出结果 ======
    print("\n=== 按 depth 统计 SFT 数据中 type 分布（按 item 数量） ===")
    for d in sorted(depth_type_counter.keys()):
        c = depth_type_counter[d]
        n0 = c.get(0, 0)
        n1 = c.get(1, 0)
        n2 = c.get(2, 0)
        other = c.get("other", 0)
        total = n0 + n1 + n2 + other
        print(f"Depth {d}:")
        print(f"  type=0: {n0}")
        print(f"  type=1: {n1}")
        print(f"  type=2: {n2}")
        if other > 0:
            print(f"  其他（type 非 0/1/2）: {other}")
        print(f"  合计 items: {total}\n")

    # 可选 CSV 导出
    if args.csv_out:
        rows = []
        for d in sorted(depth_type_counter.keys()):
            c = depth_type_counter[d]
            n0 = c.get(0, 0)
            n1 = c.get(1, 0)
            n2 = c.get(2, 0)
            other = c.get("other", 0)
            total = n0 + n1 + n2 + other
            rows.append({
                "depth": d,
                "type0": n0,
                "type1": n1,
                "type2": n2,
                "other_type": other,
                "total_items": total,
            })
        pd.DataFrame(rows).to_csv(args.csv_out, index=False, encoding="utf-8")
        print(f"\n已导出 CSV：{args.csv_out}")

if __name__ == "__main__":
    main()
