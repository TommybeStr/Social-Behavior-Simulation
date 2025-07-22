#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from collections import defaultdict

import pandas as pd

def main(processed_json, output_dir, num_parts=100):
    """
    读取已整理好的 JSON（包含 user_interests 字段），生成 SFT 训练集，
    并按总条数平分为 num_parts 个 JSON + Parquet 文件，存入 output_dir。
    """
    # 1. 确保输出目录存在且为空
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise RuntimeError(f"输出路径 {output_dir} 已存在且不是目录")
    else:
        os.makedirs(output_dir)

    # 2. 加载整理后数据
    with open(processed_json, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    # 3. 为每位作者收集潜在活跃用户（只做一次）
    potential = {}
    for post in posts:
        author = post['user']
        if author in potential:
            continue
        seen = {(post['user'], tuple(post['interests']))}
        stack = [post]
        while stack:
            node = stack.pop()
            for ch in node.get('replies', []):
                seen.add((ch['user'], tuple(ch['interests'])))
                stack.append(ch)
        potential[author] = [
            {"user_name": u, "user_interests": list(interests)}
            for u, interests in seen
        ]

    # 4. 遍历树生成 SFT 训练记录
    records = []
    for post in posts:
        stack = [(post, None, post['user'])]
        while stack:
            node, parent, root = stack.pop()
            # 历史活跃用户
            hist = []
            if parent:
                hist = [{"user_name": parent['user'],
                         "user_interests": parent['interests']}]
            # 输出列表
            outputs = [
                {"user_name": ch['user'], "content": ch['content'], "type": ch['type']}
                for ch in node.get('replies', [])
            ]
            # 添加一条训练记录
            records.append({
                "user_name": node['user'],
                "user_interests": node['interests'],
                "content": node['content'],
                "depth": node['depth'],
                "historical_interactors": hist,
                "potential_interactors": potential[root],
                "output": outputs
            })
            # 继续遍历子节点
            for ch in node.get('replies', []):
                stack.append((ch, node, root))

    # 5. 分成 num_parts 份写出
    total = len(records)
    q, r = divmod(total, num_parts)
    idx = 0

    for i in range(num_parts):
        part_size = q + (1 if i < r else 0)
        chunk = records[idx: idx + part_size]
        idx += part_size

        json_path = os.path.join(output_dir, f"part_{i+1:03d}.json")
        parquet_path = os.path.join(output_dir, f"part_{i+1:03d}.parquet")

        # 写 JSON
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(chunk, jf, ensure_ascii=False, indent=2)
        # 写 Parquet（需安装 pandas, pyarrow）
        pd.DataFrame(chunk).to_parquet(
            parquet_path,
            engine='pyarrow',
            index=False,
            compression='SNAPPY'
        )

    print(f"共写出 {num_parts} 份，每份约 {q} 条（前 {r} 份各多 1 条）到目录：{output_dir}")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python script.py processed_data.json output_folder [num_parts]")
        sys.exit(1)
    processed_json = sys.argv[1]
    output_dir     = sys.argv[2]
    parts = int(sys.argv[3]) if len(sys.argv) == 4 else 100
    main(processed_json, output_dir, parts)
