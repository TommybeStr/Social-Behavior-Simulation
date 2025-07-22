#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_sft_file.py v2.2

对超大 JSON/JSONL 训练集进行“广度十分之一”拆分，
并更新每条 user 消息里的 potential_interactors 字段，
内置流式解析，避免内存爆炸。
"""

import json
import math
import argparse
import sys
from copy import deepcopy
from collections import defaultdict

def load_samples(path):
    """
    流式加载样本：
      - JSONL 格式：逐行 parse
      - JSON 数组格式：用 ijson 流式 parse
    """
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        try:
            import ijson
        except ImportError:
            print("Error: 需要安装 ijson: pip install ijson", file=sys.stderr)
            sys.exit(1)
        with open(path, 'rb') as f:
            # 'item' 表示数组根节点下的每个元素
            for obj in ijson.items(f, 'item'):
                yield obj

def split_sample(sample, num_chunks=200):
    sys_msg = sample['messages'][0]
    # 把 messages 拆成按 depth 的 records 列表
    records = []
    msgs = sample['messages']
    i = 1
    while i < len(msgs):
        user = msgs[i]
        assistant = msgs[i+1] if i+1 < len(msgs) else None
        try:
            depth = json.loads(user['content']).get('depth', 0)
        except:
            depth = 0
        records.append({'user': deepcopy(user),
                        'assistant': deepcopy(assistant),
                        'depth': depth})
        i += 2

    # 按 depth 分组，再等分到 num_chunks 组
    depth2idxs = defaultdict(list)
    for idx, rec in enumerate(records):
        depth2idxs[rec['depth']].append(idx)

    rec2chunk = {}
    for depth, idxs in depth2idxs.items():
        total = len(idxs)
        chunk_size = max(1, math.ceil(total / num_chunks))
        for pos, ridx in enumerate(idxs):
            cid = pos // chunk_size
            if cid >= num_chunks:
                cid = num_chunks - 1
            rec2chunk[ridx] = cid

    # 收集每个 chunk 的 record idx 列表
    chunk2recs = defaultdict(list)
    for ridx, cid in rec2chunk.items():
        chunk2recs[cid].append(ridx)

    # 生成子样本
    out = []
    for cid, rec_idxs in sorted(chunk2recs.items()):
        rec_idxs.sort()
        # 先统计本 chunk 出现的所有 user_name，来过滤 potential_interactors
        included = set()
        for ridx in rec_idxs:
            ans = records[ridx]['assistant']
            if ans and ans.get('content'):
                try:
                    for a in json.loads(ans['content']):
                        included.add(a.get('user_name'))
                except:
                    pass

        new_msgs = [deepcopy(sys_msg)]
        for ridx in rec_idxs:
            rec = records[ridx]
            um = deepcopy(rec['user'])
            # 更新 potential_interactors 字段
            try:
                q = json.loads(um['content'])
                pot = q.get('potential_interactors', [])
                q['potential_interactors'] = [p for p in pot if p.get('user_name') in included]
                um['content'] = json.dumps(q, ensure_ascii=False)
            except:
                pass
            new_msgs.append(um)
            if rec['assistant']:
                new_msgs.append(deepcopy(rec['assistant']))

        new_id = f"{sample['id']}_chunk{cid}"
        out.append({'id': new_id, 'messages': new_msgs})
    return out

def main():
    parser = argparse.ArgumentParser(
        description="按广度十分之一拆分 SFT 样本，并更新 potential_interactors；支持流式大文件解析"
    )
    parser.add_argument('--input', '-i', required=True, help="输入 JSON 或 JSONL 文件路径")
    parser.add_argument('--output', '-o', required=True, help="输出 JSONL 文件路径")
    parser.add_argument('--chunks', '-n', type=int, default=10, help="每层拆分组数，默认20")
    args = parser.parse_args()

    total_in = 0
    total_out = 0
    with open(args.output, 'w', encoding='utf-8') as fout:
        for sample in load_samples(args.input):
            total_in += 1
            for sub in split_sample(sample, num_chunks=args.chunks):
                fout.write(json.dumps(sub, ensure_ascii=False) + '\n')
                total_out += 1

    print(f"处理完毕：输入样本 {total_in} 条，输出子样本 {total_out} 条。")

if __name__ == '__main__':
    main()