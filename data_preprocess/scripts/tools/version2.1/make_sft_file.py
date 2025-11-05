#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单轮 SFT 构造脚本（两层分类版，user 插入 <TL?>，并新增 assistant 标注 JSON）
- 只对 depth 0（root）和 depth 1 的节点产样本：
  * depth=0 → target_layer=0 → 用分类头0（预测第一层）
  * depth=1 → target_layer=1 → 用分类头1（预测第二层）
- 每条样本 messages：system + user + assistant
  * user 文末包含 <TL0>/<TL1>
  * assistant 为覆盖候选数组的唯一 JSON 输出；content 对评论/转发加 <|cstart|>…<|cend|> 外壳
- 统计&过滤：
  1) 候选池规模（均值、min/max、阈值计数）
  2) 样本构造时跳过候选池人数 > 阈值(默认1000)的根作者
  3) 分 depth 报告：平均每条样本从多少候选中选出多少 gold（gold=type!=0）
  4) 验证集示例：输出 10 条 depth=1(≈depth2) 且有 gold 的样本 ID
"""

import os
import json
import argparse
import random
import re
from collections import defaultdict, deque
from typing import List, Dict, Any, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------- 常量 ----------
PSEP_TOKEN = "<|psep|>"
PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END   = "\n</POTENTIAL_SPANS>\n"
CSTART_TOKEN = "<|cstart|>"
CEND_TOKEN   = "<|cend|>"

# ---------- System Prompt ----------
SYSTEM_PROMPT = f'''你是社交媒体互动预测专家。请严格依据 user 消息中的标注字段进行判断，并输出一个覆盖全部候选的 JSON 数组（顺序必须严格与候选顺序一致）。

【【输入字段（单样本 JSON）】
- username：作者
- interests：作者兴趣（数组）
- content：正文文本。
- historicalinteractors：与作者历史上发生过互动的用户名列表。注意：其末尾会追加一个特殊段落 `<POTENTIAL_SPANS>`，用于提供候选人信息。

【关于 <POTENTIAL_SPANS>】
- `<POTENTIAL_SPANS>` 紧跟在historicalinteractors末尾，并以 `</POTENTIAL_SPANS>` 结束（严格成对）。
- 其中每个候选用成对分隔符 `{PSEP_TOKEN}` 包裹：`{PSEP_TOKEN}{{候选JSON}}{PSEP_TOKEN}`。
- 候选 JSON 严格包含：{{"user_name": 候选人, "interests": 候选人兴趣, "depth": 层级}}。
- 这些候选块的先后顺序即为评分类与输出顺序的唯一依据；禁止重排、丢失或增添。

【唯一输出（严格格式）】
- 输出一个 JSON 数组，长度等于候选数量，顺序与 <POTENTIAL_SPANS> 中候选顺序一致。
- 数组元素结构：
  {{"user_name":"...", "content":"{CSTART_TOKEN}...{CEND_TOKEN}", "type":0/1/2}}
  - type：0=无互动；1=评论；2=转发微博
  - content：type=1/2 时用 {CSTART_TOKEN}…{CEND_TOKEN} 包裹（可为空但标记必须存在）；type=0 时输出 "{CSTART_TOKEN}{CEND_TOKEN}"。
- 仅输出该 JSON 数组，不得包含解释或多余文本。
- 禁止使用 {CSTART_TOKEN}/{CEND_TOKEN} 之外的自造标记。
'''

# ---------- Arrow schema ----------
MSG_STRUCT = pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
    pa.field("loss", pa.int64()).with_nullable(True),
])
TABLE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("messages", pa.list_(MSG_STRUCT)),
    pa.field("targets_per_potential_types", pa.list_(pa.int32())),
    pa.field("targets_comment_texts", pa.list_(pa.string())),
    pa.field("node_depth", pa.int32()),   # 0/1，与 <TL?> 一致
    pa.field("sft_chunk_info", pa.large_string()),
])

def rows_to_arrow_table(rows: List[Dict[str, Any]]) -> pa.Table:
    ids = pa.array([r['id'] for r in rows], type=pa.string())
    msgs = pa.array(
        [[{"role": m.get("role"), "content": m.get("content"), "loss": m.get("loss")} for m in r['messages']]
         for r in rows],
        type=pa.list_(MSG_STRUCT)
    )
    t_types  = pa.array([r.get("targets_per_potential_types", []) for r in rows], type=pa.list_(pa.int32()))
    t_comms  = pa.array([r.get("targets_comment_texts", []) for r in rows], type=pa.list_(pa.string()))
    node_d   = pa.array([int(r.get("node_depth", 0)) for r in rows], type=pa.int32())
    infos = pa.array([json.dumps(r.get('sft_chunk_info', {}), ensure_ascii=False) for r in rows], type=pa.large_string())
    return pa.Table.from_arrays([ids, msgs, t_types, t_comms, node_d, infos], schema=TABLE_SCHEMA)

def save_parquet_rows(rows: List[Dict[str, Any]], path: str, *, batch_size: int = 4000, desc: str = "写 Parquet", compression: str = "zstd"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        empty = pa.Table.from_arrays(
            [pa.array([], type=pa.string()),
             pa.array([], type=pa.list_(MSG_STRUCT)),
             pa.array([], type=pa.list_(pa.int32())),
             pa.array([], type=pa.list_(pa.string())),
             pa.array([], type=pa.int32()),
             pa.array([], type=pa.large_string())],
            schema=TABLE_SCHEMA
        )
        pq.write_table(empty, path, compression=compression); return
    writer = None
    total_batches = (len(rows) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc=desc) as pbar:
        for i in range(0, len(rows), batch_size):
            table = rows_to_arrow_table(rows[i:i+batch_size])
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression, use_dictionary=True)
            writer.write_table(table); pbar.update(1)
    if writer is not None:
        writer.close()

def iter_tree_nodes(root: Dict[str, Any]):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in (node.get('replies') or []):
            stack.append(child)

def _safe_depth(node: Dict[str, Any]) -> int:
    try:
        return int(node.get('depth', 0))
    except Exception:
        return 0

# ---------- 候选池 ----------
def build_user_interest_map_and_group_by_root(records: List[Dict[str, Any]]):
    user_interest_map: Dict[str, Any] = {}
    root_user_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in tqdm(records, desc="扫描记录并构建画像/根作者分组"):
        root_user = str(record.get('user') or "")
        if root_user:
            root_user_to_records[root_user].append(record)
        for node in iter_tree_nodes(record):
            u = str(node.get('user') or "")
            ints = node.get('interests', [])
            if u and ints and (u not in user_interest_map):
                user_interest_map[u] = ints
    return user_interest_map, root_user_to_records

def collect_candidate_pool_for_root(records_of_root: List[Dict[str, Any]], user_interest_map: Dict[str, Any]) -> List[str]:
    users: Set[str] = set()
    for rec in records_of_root:
        for node in iter_tree_nodes(rec):
            u = str(node.get('user') or "")
            if u and (u in user_interest_map) and user_interest_map[u]:
                users.add(u)
    return list(users)

# ---------- 文本处理 ----------
def _strip_retweet_tail(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    idx1 = text.find("//@"); idx2 = text.find("/@")
    idxs = [i for i in (idx1, idx2) if i != -1]
    if not idxs: return text.strip()
    return text[:min(idxs)].rstrip()

def _sanitize_content_for_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace(CSTART_TOKEN, "").replace(CEND_TOKEN, "").strip()

# ---------- <POTENTIAL_SPANS> ----------
def make_potential_span_text(potentials: List[str], user_interest_map: Dict[str, Any], target_layer: int) -> str:
    parts = [PSEP_BLOCK_START]
    dval = int(target_layer)
    for pid in potentials:
        block = {"user_name": pid, "interests": user_interest_map.get(pid, []) or [], "depth": dval}
        parts.append(PSEP_TOKEN)
        parts.append(json.dumps(block, ensure_ascii=False, separators=(',', ':')))
        parts.append(PSEP_TOKEN)
    parts.append(PSEP_BLOCK_END)
    return "".join(parts)

def make_user_plain_text_with_cached_spans(node: Dict[str, Any], ancestors: List[str], cached_spans_text: str, target_layer: int) -> str:
    base_content = _strip_retweet_tail(node.get('content') or "")
    username = str(node.get('user') or "")
    userinterest = node.get('interests', [])
    hist_ids = [str(a) for a in ancestors]

    user_plain_text = (
        "username: " + username + "\n" +
        "content:\n" + base_content + "\n" +
        "userinterest: " + json.dumps(userinterest, ensure_ascii=False) + "\n" +
        "historicalinteractors: " + json.dumps(hist_ids, ensure_ascii=False) + "\n" +
        "potentialspan:" + cached_spans_text +
        f"<TL{int(target_layer)}>"
    )
    if not user_plain_text.endswith(f"<TL{int(target_layer)}>"):
        raise RuntimeError("BUG: <TL?> not at end")
    return user_plain_text

# ---------- 标签构造 ----------
def get_comment_or_repost(child: Dict[str, Any]) -> Tuple[int, str]:
    raw_content = _strip_retweet_tail(child.get('content') or "")
    raw_type = str(child.get('type') or "评论")
    mapped_type = 2 if raw_type == "转发微博" else 1
    if mapped_type == 1 and ("//@") in (child.get('content') or ""):
        mapped_type = 2
    safe = _sanitize_content_for_markers(raw_content)
    return mapped_type, safe

def extract_children_map(root: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    node_children: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in iter_tree_nodes(root):
        for child in (node.get('replies') or []):
            node_children[id(node)].append(child)
    return node_children

def build_child_map(children: List[Dict[str, Any]]) -> Dict[str, Tuple[int, str]]:
    m = {}
    for c in (children or []):
        u = str(c.get('user') or "").strip()
        if not u:
            continue
        t, txt = get_comment_or_repost(c)
        m[u] = (t, txt if t == 1 else "")
    return m

# ---------- 分块 ----------
def chunk_list_no_overlap(lst: List[str], k: int) -> List[List[str]]:
    if k <= 0:
        return [lst] if lst else []
    return [lst[i:i+k] for i in range(0, len(lst), k)]

# ---------- 生成样本 ----------
def generate_single_turn_rows_for_record(
    record_root: Dict[str, Any],
    root_user: str,
    root_potential_full: List[str],
    user_interest_map: Dict[str, Any],
    k_per_chunk: int,
    spans_cache: Dict[Tuple[str, int, int], str],  # (root_user, chunk_idx, target_layer)
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root_potential_full:
        return rows

    root_chunks = chunk_list_no_overlap(root_potential_full, k_per_chunk)
    node_children = extract_children_map(record_root)

    queue = deque([(record_root, [])])
    while queue:
        node, ancestors = queue.popleft()
        d = _safe_depth(node)

        if d in (0, 1):
            target_layer = 0 if d == 0 else 1
            children = node_children.get(id(node), [])
            cm = build_child_map(children)

            for ch_idx, pot_chunk in enumerate(root_chunks):
                cache_key = (root_user, ch_idx, target_layer)
                if cache_key in spans_cache:
                    spans_text = spans_cache[cache_key]
                else:
                    spans_text = make_potential_span_text(pot_chunk, user_interest_map, target_layer)
                spans_cache[cache_key] = spans_text

                # user 文本（含 <TL?>）
                user_plain = make_user_plain_text_with_cached_spans(node, ancestors, spans_text, target_layer)

                # 构造 assistant JSON（长度与 pot_chunk 一致；顺序严格一致）
                types, comms = [], []
                out_arr = []
                for pid in pot_chunk:
                    if pid in cm:
                        t, txt = cm[pid]
                    else:
                        t, txt = 0, ""
                    types.append(t)
                    comms.append(txt)
                    if t == 1:
                        content = f"{CSTART_TOKEN}{txt}{CEND_TOKEN}"
                    else:
                        content = f"{CSTART_TOKEN}{CEND_TOKEN}"
                    out_arr.append({"user_name": pid, "content": content, "type": int(t)})

                assistant_text = json.dumps(out_arr, ensure_ascii=False)

                messages = [
                    {'role': 'system',    'content': SYSTEM_PROMPT, 'loss': 0},
                    {'role': 'user',      'content': user_plain,    'loss': 0},
                    {'role': 'assistant', 'content': assistant_text,'loss': 1},
                ]

                rows.append({
                    'id': "{}_node_{}_potchunk_{}_tL{}".format(str(record_root.get('id') or ''), id(node), ch_idx, target_layer),
                    'messages': messages,
                    'targets_per_potential_types': types,
                    'targets_comment_texts': comms,
                    'node_depth': int(target_layer),
                    'sft_chunk_info': {
                        "record_id": str(record_root.get('id') or ''),
                        "root_user": root_user,
                        "node_user": str(node.get('user') or ''),
                        "orig_node_depth": d,
                        "target_layer": target_layer,
                        "potential_chunk_index": ch_idx,
                        "k_per_chunk": k_per_chunk,
                        "format": "single_turn_messages_with_targets_and_assistant_json",
                        "psep_token": PSEP_TOKEN,
                        "cstart_token": CSTART_TOKEN,
                        "cend_token": CEND_TOKEN,
                        "psep_block_start": PSEP_BLOCK_START.strip(),
                        "psep_block_end": PSEP_BLOCK_END.strip(),
                    }
                })

        for child in node_children.get(id(node), []):
            queue.append((child, ancestors + [str(node.get('user') or '')]))

    return rows

# ---------- 分 depth 统计（新增） ----------
def report_depth_candidate_gold_stats(rows: List[Dict[str, Any]]):
    # gold 定义：type != 0（评论/转发都是正例）
    acc = {
        0: {"samples": 0, "cand_total": 0, "gold_total": 0},
        1: {"samples": 0, "cand_total": 0, "gold_total": 0},
    }
    for r in rows:
        d = int(r.get("node_depth", 0))
        if d not in acc:
            continue
        types = r.get("targets_per_potential_types", []) or []
        cand = len(types)
        gold = sum(1 for t in types if int(t) != 0)
        acc[d]["samples"] += 1
        acc[d]["cand_total"] += cand
        acc[d]["gold_total"] += gold

    print("[stats] 分 depth 的候选与 gold 平均统计（gold=type!=0）：")
    for d in (0, 1):
        s = acc[d]["samples"]
        ct = acc[d]["cand_total"]
        gt = acc[d]["gold_total"]
        if s > 0:
            avg_cand = ct / s
            avg_gold = gt / s
            ratio = (gt / ct) if ct > 0 else 0.0
            print(f"        depth={d}: 样本数={s}, 平均候选={avg_cand:.2f}, 平均gold={avg_gold:.2f}, gold/候选={ratio:.4f}")
        else:
            print(f"        depth={d}: 样本数=0 (无统计)")

# ---------- 入口 ----------
_ID_RE = re.compile(r"^(?P<rec>.+?)_root_(?P<root>.+?)_flchunk_(?P<idx>\d+)$")
def parse_record_id(sample_id: str) -> str:
    m = _ID_RE.match(str(sample_id)); return m.group("rec") if m else ""

def main(input_file: str,
         output_dir: str,
         shuffle_seed: int = 42,
         parquet_batch_size: int = 4000,
         k_potential_per_chunk: int = 100,
         parquet_compression: str = "zstd",
         large_pool_threshold: int = 1000,
         val_depth2_head: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin:
        records = json.load(fin)
    if not isinstance(records, list):
        raise ValueError("输入 JSON 顶层必须是 list")

    user_interest_map, root_user_to_records = build_user_interest_map_and_group_by_root(records)

    root_user_to_candidate_pool: Dict[str, List[str]] = {}
    for root_user, recs in tqdm(root_user_to_records.items(), total=len(root_user_to_records), desc="按根作者构建候选池"):
        pool = collect_candidate_pool_for_root(recs, user_interest_map)
        rng_local = random.Random(shuffle_seed ^ (hash(root_user) & 0xffffffff))
        rng_local.shuffle(pool)
        root_user_to_candidate_pool[root_user] = pool

    # === 候选池规模统计（按根作者） ===
    pool_sizes = [len(v) for v in root_user_to_candidate_pool.values()]
    if pool_sizes:
        avg_sz = sum(pool_sizes) / len(pool_sizes)
        n_large = sum(1 for s in pool_sizes if s > large_pool_threshold)
        print("[stats] 候选池（按根作者）规模统计：")
        print(f"        根作者数 = {len(pool_sizes)}")
        print(f"        平均每池人数 = {avg_sz:.2f}")
        print(f"        min/max = {min(pool_sizes)}/{max(pool_sizes)}")
        print(f"        候选总人数（各根作者池大小之和，非全局去重）= {sum(pool_sizes)}")
        print(f"        候选池 > {large_pool_threshold} 的根作者数 = {n_large}")
    else:
        print("[stats] 未构建到任何候选池（root_user_to_candidate_pool 为空）。")

    # === 跳过候选池人数 > 阈值 的根作者 ===
    skipped_roots = [ru for ru, pool in root_user_to_candidate_pool.items() if len(pool) > large_pool_threshold]
    skipped_records = sum(len(root_user_to_records.get(ru, [])) for ru in skipped_roots)
    if skipped_roots:
        print(f"[filter] 将在样本构造中跳过候选池 > {large_pool_threshold} 的根作者：{len(skipped_roots)} 个（涉及记录 {skipped_records} 条）")
    else:
        print("[filter] 无需跳过任何根作者。")

    rng = random.Random(shuffle_seed)

    rows: List[Dict[str, Any]] = []
    spans_cache: Dict[Tuple[str, int, int], str] = {}

    total_rec_for_progress = sum(len(recs) for recs in root_user_to_records.values())
    with tqdm(total=total_rec_for_progress, desc="按记录生成【单轮/两层】样本(含缓存)") as pbar:
        for root_user, recs in root_user_to_records.items():
            root_pool_full = root_user_to_candidate_pool.get(root_user, [])
            # 跳过大池根作者；进度条补齐
            if len(root_pool_full) > large_pool_threshold:
                pbar.update(len(recs))
                continue
            for rec in recs:
                rec_rows = generate_single_turn_rows_for_record(
                    record_root=rec,
                    root_user=root_user,
                    root_potential_full=root_pool_full,
                    user_interest_map=user_interest_map,
                    k_per_chunk=k_potential_per_chunk,
                    spans_cache=spans_cache,
                )
                rows.extend(rec_rows)
                pbar.update(1)

    # === 分 depth 报告每条样本平均候选/平均 gold ===
    report_depth_candidate_gold_stats(rows)

    # record_id 级切分
    df_idx = pd.DataFrame({"idx": list(range(len(rows)))})
    df_idx["record_id"] = [rows[i]["sft_chunk_info"]["record_id"] for i in range(len(rows))]

    unique_recs = list(dict.fromkeys(df_idx["record_id"].fillna("").astype(str).tolist()))
    rng.shuffle(unique_recs)
    n_rec = len(unique_recs)
    n_train_rec = int(round(n_rec * 0.85))
    n_val_rec   = int(round(n_rec * 0.05))
    n_test_rec  = max(0, n_rec - n_train_rec - n_val_rec)

    recs_train = set(unique_recs[:n_train_rec])
    recs_val   = set(unique_recs[n_train_rec:n_train_rec + n_val_rec])
    recs_test  = set(unique_recs[n_train_rec + n_val_rec:])

    idx_train = df_idx[df_idx["record_id"].isin(recs_train)]["idx"].tolist()
    idx_val   = df_idx[df_idx["record_id"].isin(recs_val)]["idx"].tolist()
    idx_test  = df_idx[df_idx["record_id"].isin(recs_test)]["idx"].tolist()

    rows_train = [rows[i] for i in idx_train]
    rows_val   = [rows[i] for i in idx_val]
    rows_test  = [rows[i] for i in idx_test]

    # === 验证集示例：输出 depth=1(≈depth2) 且有 gold 的样本 ID（最多 N 条） ===
    # 在本脚本中 node_depth=1 即“预测第二层”，对应你的“depth2”
    val_depth2_ids = []
    for r in rows_val:
        if int(r.get("node_depth", 0)) != 1:
            continue
        types = r.get("targets_per_potential_types", []) or []
        if any(int(t) != 0 for t in types):
            val_depth2_ids.append(str(r.get("id")))
    # 去重并截取
    val_depth2_ids = list(dict.fromkeys(val_depth2_ids))[:max(0, int(val_depth2_head))]
    if val_depth2_ids:
        print(f"[val] depth=1(≈depth2) 且有 gold 的样本ID（最多 {val_depth2_head} 条）：")
        for i, sid in enumerate(val_depth2_ids, 1):
            print(f"       {i:02d}. {sid}")
    else:
        print("[val] 未找到 depth=1(≈depth2) 且有 gold 的验证集样本。")

    print("[split] record_id 级：train/val/test 记录数 = {}/{}/{}".format(len(recs_train), len(recs_val), len(recs_test)))
    print("[split] 样本条数：train/val/test = {}/{}/{}".format(len(rows_train), len(rows_val), len(rows_test)))

    train_out = os.path.join(output_dir, "train.parquet")
    val_out   = os.path.join(output_dir, "val.parquet")
    test_out  = os.path.join(output_dir, "test.parquet")

    save_parquet_rows(rows_train, train_out, batch_size=parquet_batch_size, desc="写 Parquet(train)", compression=parquet_compression)
    save_parquet_rows(rows_val,   val_out,   batch_size=parquet_batch_size, desc="写 Parquet(val)",   compression=parquet_compression)
    save_parquet_rows(rows_test,  test_out,  batch_size=parquet_batch_size, desc="写 Parquet(test)",  compression=parquet_compression)

    print("[done] 单轮/两层 SFT 构造完成（已输出 3 个 Parquet）")
    print("train : {}\nval   : {}\ntest  : {}".format(train_out, val_out, test_out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单轮 SFT 构造（两层分类版；record_id 级切分 + 输出 3 个 Parquet）')
    parser.add_argument('--input', required=True, help='输入 JSON 文件路径（原始树形数据，顶层为 list）')
    parser.add_argument('--output_dir', required=True, help='输出文件夹（将生成 train/val/test 三个 Parquet）')
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--parquet_batch_size', type=int, default=4000)
    parser.add_argument('--k_potential_per_chunk', type=int, default=100, help='每条样本放入的候选个数 K')
    parser.add_argument('--parquet_compression', type=str, default="zstd", help='Parquet 压缩算法（zstd/snappy/uncompressed 等）')
    parser.add_argument('--large_pool_threshold', type=int, default=1000, help='阈值：候选池人数 > 阈值 的根作者将被跳过')
    parser.add_argument('--val_depth2_head', type=int, default=30, help='在验证集中输出的 depth=1(≈depth2) 且有 gold 的样本ID数量上限')
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        parquet_batch_size=args.parquet_batch_size,
        k_potential_per_chunk=args.k_potential_per_chunk,
        parquet_compression=args.parquet_compression,
        large_pool_threshold=args.large_pool_threshold,
        val_depth2_head=args.val_depth2_head,
    )
