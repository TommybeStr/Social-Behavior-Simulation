#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
适配你当前 SFT 构造脚本产出的 Parquet 的 depth 统计脚本

统计项：
- 按 depth 统计回复类型（评论 vs 转发微博）
- 检查 type=2 时 content 是否含转发特征（简单同义词判定）
- 检查 type=1 但 content=转发微博 的异常
- 统计 assistant 响应中：无互动 / 评论 / 转发微博 的占比（按样本）

数据假设（来自你的构造脚本）：
- 列 schema 至少包含：id, messages（list<struct<role:string, content:string, loss:int64>>）
- user.content 是纯文本，形如：
    username: <name>
    content:
    <text>
    userinterest: [...]
    historicalinteractors: [...]
    potentialspan:
    <POTENTIAL_SPANS>
      <|psep|>{"user_name":"...","interests":[...],"depth":d}<|psep|> ...
    </POTENTIAL_SPANS>
- assistant.content 是严格的 JSON 数组字符串，形如：
    [{"user_name":"...","content":"<|cstart|>...<|cend|>","type":0/1/2}, ...]
"""

import os
import re
import json
import glob
import argparse
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

# ====== 常量 / 正则 ======
_PSEP_TOKEN = "<|psep|>"
_PSEP_BLOCK_RE = re.compile(r"<POTENTIAL_SPANS>\s*(.*?)\s*</POTENTIAL_SPANS>", re.DOTALL)
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RT_SYNONYMS = {"转发微博", "转发", "repost", "retweet", "share"}

def content_is_retweet(v: Any) -> bool:
    if not isinstance(v, str): return False
    return v.strip().lower() in _RT_SYNONYMS

def safe_json_loads(s: Any):
    if not isinstance(s, str): return None
    try: return json.loads(s)
    except Exception: return None

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
            seen.add(p); out.append(p)
    return out

def iter_parquet(paths: List[str]) -> Iterable[Tuple[str, pd.DataFrame]]:
    for p in paths:
        df = pd.read_parquet(p, engine="pyarrow")
        # 仅保留可能用到的列；若缺失也不强求
        keep = [c for c in ["id","messages"] if c in df.columns]
        if not keep:
            keep = list(df.columns)
        yield p, df[keep].copy()

def to_list(obj: Any) -> List[Any]:
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, list): return obj
    if isinstance(obj, tuple): return list(obj)
    if isinstance(obj, np.ndarray):
        try: return obj.tolist()
        except Exception: pass
    try:
        import pyarrow as pa
        if isinstance(obj, (pa.Array, pa.ChunkedArray, pa.ListScalar)):
            return obj.to_pylist()
    except Exception:
        pass
    j = safe_json_loads(obj)
    return j if isinstance(j, list) else []

# ====== 解析 user / assistant ======
def parse_username_from_user_plain(s: Any) -> str:
    if not isinstance(s, str): return ""
    m = _USERNAME_LINE_RE.search(s)
    return (m.group("name").strip() if m else "")

def parse_depth_from_user_plain(s: Any) -> int:
    """
    你的构造逻辑：<POTENTIAL_SPANS> 中每个候选 JSON 都携带了 "depth": 当前 user 节点的层级。
    所以这里取这个 block 里第一个 JSON 的 depth 即可。
    """
    if not isinstance(s, str): return 0
    m = _PSEP_BLOCK_RE.search(s)
    if not m: return 0
    body = m.group(1)
    # 在 body 里找第一个 {...} JSON
    # 简单做法：按 <|psep|> 分割取第一个 JSON
    parts = body.split(_PSEP_TOKEN)
    for part in parts:
        part = part.strip()
        if not part or not part.startswith("{"): continue
        obj = safe_json_loads(part)
        if isinstance(obj, dict) and "depth" in obj:
            try: return int(obj.get("depth", 0))
            except Exception: return 0
    return 0

def parse_assistant_items(assistant_content: Any) -> List[Dict[str, Any]]:
    # 严格 JSON 数组字符串；但也兼容已是 list 或 dict 包 items 的情况
    if isinstance(assistant_content, list):
        return [it for it in assistant_content if isinstance(it, dict)]
    if isinstance(assistant_content, dict):
        for key in ("items","output","candidates","data","list","arr"):
            if key in assistant_content and isinstance(assistant_content[key], list):
                return [it for it in assistant_content[key] if isinstance(it, dict)]
        if any(k in assistant_content for k in ("type","user_name","username","name")):
            return [assistant_content]
        return []
    arr = safe_json_loads(assistant_content)
    if isinstance(arr, list):
        return [it for it in arr if isinstance(it, dict)]
    if isinstance(arr, dict):
        return parse_assistant_items(arr)
    return []

def get_item_user_name(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict): return ""
    for k in ["user_name","username","user","name","screen_name"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser(description="按 depth 统计（适配你当前 SFT Parquet 格式）")
    ap.add_argument("--inputs", nargs="+", required=True, help="Parquet 路径/目录/通配符，可多个")
    ap.add_argument("--min_depth", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--csv_out", type=str, default="")
    args = ap.parse_args()

    paths = collect_paths(args.inputs)
    if not paths:
        print("未找到任何 Parquet；请检查 --inputs。")
        return

    # 主统计：按 depth 计数（评论 / 转发微博）
    depth_counters: Dict[int, Counter] = defaultdict(Counter)

    # 转发微博细分
    rt_type_total: Dict[int, int] = defaultdict(int)
    rt_content_is_rt: Dict[int, int] = defaultdict(int)

    # 统计 A：转发微博项后续是否有人互动（看下一层是否出现该 child 成为 user）
    A_rt_ct_isrt_total: Dict[int, int] = defaultdict(int)
    A_rt_ct_isrt_has_engage: Dict[int, int] = defaultdict(int)
    A_rt_ct_not_isrt_total: Dict[int, int] = defaultdict(int)
    A_rt_ct_not_isrt_has_engage: Dict[int, int] = defaultdict(int)

    # 统计 B：type=评论 但 content=转发微博
    B_comment_content_isrt_by_depth: Dict[int, int] = defaultdict(int)
    B_comment_total_by_depth: Dict[int, int] = defaultdict(int)

    # 统计 C：assistant 响应类型占比（按样本）
    assistant_sample_type_counter: Dict[int, Counter] = defaultdict(Counter)

    for p, df in iter_parquet(paths):
        print(f"[INFO] 处理文件: {p}  rows={len(df)}")

        # 收集 (depth, user_name) 出现表，用于 A 统计“后续是否有人互动”
        # 注意：一个样本(messages)中包含多个 user/assistant 对；我们逐行展开
        user_occurs: set[Tuple[int, str]] = set()
        # user 块列表（depth, username, assistant_items）
        user_blocks: List[Tuple[int, str, List[Dict[str, Any]]]] = []

        for _, row in df.iterrows():
            msgs = to_list(row.get("messages"))
            if not msgs: continue

            i = 0
            while i < len(msgs):
                m = msgs[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    user_plain = m.get("content", "")
                    # 从纯文本中解析 depth / username
                    user_depth = parse_depth_from_user_plain(user_plain)
                    user_name  = parse_username_from_user_plain(user_plain)

                    # depth 过滤
                    if user_depth < args.min_depth or (args.max_depth >= 0 and user_depth > args.max_depth):
                        # 跳过其相邻 assistant
                        i += 2 if (i+1 < len(msgs) and isinstance(msgs[i+1], dict) and msgs[i+1].get("role")=="assistant") else 1
                        continue

                    # 相邻 assistant
                    items: List[Dict[str, Any]] = []
                    if i + 1 < len(msgs) and isinstance(msgs[i+1], dict) and msgs[i+1].get("role") == "assistant":
                        items = parse_assistant_items(msgs[i+1].get("content", ""))

                    user_blocks.append((user_depth, user_name, items))
                    if user_name:
                        user_occurs.add((user_depth, user_name))
                    i += 2  # 跳过 assistant
                else:
                    i += 1

        # —— 统计 C（样本级） —— #
        for d, uname, arr in user_blocks:
            if not arr:
                assistant_sample_type_counter[d]["无互动"] += 1
            else:
                has_comment = any(int(it.get("type", 0)) == 1 for it in arr)
                has_rt      = any(int(it.get("type", 0)) == 2 for it in arr)
                if has_rt:
                    assistant_sample_type_counter[d]["转发微博"] += 1
                elif has_comment:
                    assistant_sample_type_counter[d]["评论"] += 1
                else:
                    assistant_sample_type_counter[d]["无互动"] += 1

        # —— 主统计 + A/B —— #
        for d, uname, arr in user_blocks:
            for it in arr:
                typ = int(it.get("type", 0))
                typ_str = "评论" if typ == 1 else ("转发微博" if typ == 2 else "无互动")

                if typ_str in {"评论", "转发微博"}:
                    depth_counters[d][typ_str] += 1

                    if typ_str == "转发微博":
                        rt_type_total[d] += 1
                        if content_is_retweet(it.get("content")):
                            rt_content_is_rt[d] += 1

                        # 统计 A：该 child 是否在 d+1 层出现过（认为有后续互动）
                        child_name = get_item_user_name(it)
                        if child_name:
                            has_engage = ((d+1, child_name) in user_occurs)
                            if content_is_retweet(it.get("content")):
                                A_rt_ct_isrt_total[d] += 1
                                if has_engage:
                                    A_rt_ct_isrt_has_engage[d] += 1
                            else:
                                A_rt_ct_not_isrt_total[d] += 1
                                if has_engage:
                                    A_rt_ct_not_isrt_has_engage[d] += 1

                if typ_str == "评论":
                    B_comment_total_by_depth[d] += 1
                    if content_is_retweet(it.get("content")):
                        B_comment_content_isrt_by_depth[d] += 1

    # ====== 输出 ======
    print("\n=== 主要统计结果 ===")

    print("\n[按 depth 统计回复类型（仅统计实际发生的互动）]")
    for d in sorted(depth_counters.keys()):
        c = depth_counters[d]
        n_comment = c.get("评论", 0)
        n_rt = c.get("转发微博", 0)
        total_interactions = n_comment + n_rt
        print(f"Depth {d}:")
        print(f"  评论: {n_comment}")
        print(f"  转发微博: {n_rt}")
        print(f"  互动总数: {total_interactions}")
        if n_rt > 0:
            k = rt_content_is_rt.get(d, 0)
            print(f"  转发微博中content也为转发微博: {k} ({k/n_rt*100:.2f}%)")
        print()

    print("\n[统计 A：转发微博项后续是否有人互动]")
    all_d = sorted(set(A_rt_ct_isrt_total.keys()) | set(A_rt_ct_not_isrt_total.keys()))
    for d in all_d:
        print(f"Depth {d}:")
        t1 = A_rt_ct_isrt_total.get(d, 0)
        h1 = A_rt_ct_isrt_has_engage.get(d, 0)
        if t1 > 0:
            print(f"  转发微博(content=转发微博): 总数{t1}, 有后续互动{h1} ({h1/t1*100:.2f}%)")
        t0 = A_rt_ct_not_isrt_total.get(d, 0)
        h0 = A_rt_ct_not_isrt_has_engage.get(d, 0)
        if t0 > 0:
            print(f"  转发微博(content!=转发微博): 总数{t0}, 有后续互动{h0} ({h0/t0*100:.2f}%)")

    print("\n[统计 B：type=评论 且 content=转发微博]")
    for d in sorted(B_comment_total_by_depth.keys()):
        tot = B_comment_total_by_depth[d]
        bad = B_comment_content_isrt_by_depth.get(d, 0)
        if tot > 0:
            print(f"Depth {d}: 评论总数{tot}, 其中content=转发微博的{bad} ({bad/tot*100:.2f}%)")

    print("\n[统计 C：assistant 响应类型占比（按样本）]")
    for d in sorted(assistant_sample_type_counter.keys()):
        c = assistant_sample_type_counter[d]
        total = sum(c.values())
        if total > 0:
            print(f"Depth {d} (样本数={total}):")
            print(f"  无互动: {c.get('无互动', 0)/total*100:.2f}%")
            print(f"  评论: {c.get('评论', 0)/total*100:.2f}%")
            print(f"  转发微博: {c.get('转发微博', 0)/total*100:.2f}%")

    # 可选 CSV
    if args.csv_out:
        rows = []
        for d in sorted(depth_counters.keys()):
            c = depth_counters[d]
            n_comment = c.get("评论", 0)
            n_rt = c.get("转发微博", 0)
            total_interactions = n_comment + n_rt
            rt_total = rt_type_total.get(d, 0)
            rt_rt = rt_content_is_rt.get(d, 0)
            pct = f"{(rt_rt/rt_total*100):.2f}%" if rt_total else "0.00%"
            rows.append({
                "depth": d,
                "评论": n_comment,
                "转发微博": n_rt,
                "互动总数": total_interactions,
                "转发微博中content也为转发微博": rt_rt,
                "转发微博content=转发微博占比": pct
            })
        pd.DataFrame(rows).to_csv(args.csv_out, index=False, encoding="utf-8")
        print(f"\n已导出 CSV：{args.csv_out}")

if __name__ == "__main__":
    main()
