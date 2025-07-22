#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_tokens_and_save_to_csv.py

统计 JSONL 文件中每一行的 token 总数，并将结果存入 CSV 文件。
"""

import json
import argparse
import csv
from transformers import AutoTokenizer

def compute_token_length(text: str, tokenizer) -> int:
    """
    计算文本的 token 数。
    """
    if not text:  # 如果 text 是 None 或空字符串
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def main(jsonl_path: str, model_path: str, output_csv: str):
    """
    统计 JSONL 文件每一行的 token 总数，并将结果存入 CSV 文件。
    """
    # 初始化 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"初始化 tokenizer 失败: {e}")
        return

    # 打开 CSV 文件，准备写入结果
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入 CSV 表头
        csv_writer.writerow(['行号', '总 Token 数'])

        line_count = 0

        # 逐行读取 JSONL 文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)  # 解析 JSON 行
                except json.JSONDecodeError:
                    print(f"跳过解析错误的行: {line.strip()}")
                    continue

                # 计算当前行中所有 messages 的 token 总数
                line_total_tokens = 0
                for msg in sample.get('messages', []):
                    if not isinstance(msg, dict):  # 确保 msg 是字典
                        continue
                    line_total_tokens += compute_token_length(msg.get('content', ''), tokenizer)

                # 将行号和 token 数写入 CSV 文件
                csv_writer.writerow([line_count + 1, line_total_tokens])

                # 打印当前行的结果（可选）
                #print(f"第 {line_count + 1} 行的总 token 数 = {line_total_tokens}")

                line_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="统计 JSONL 文件每一行的 token 总数，并将结果存入 CSV 文件"
    )
    parser.add_argument(
        '--jsonl_path',
        help='输入 JSONL 文件路径（.jsonl）'
    )
    parser.add_argument(
        '--model_path',
        help='transformers 模型或 tokenizer 路径，如 /home/zss/.../bert-base-uncased'
    )
    parser.add_argument(
        '--output_csv',
        help='输出 CSV 文件路径'
    )
    args = parser.parse_args()
    main(args.jsonl_path, args.model_path, args.output_csv)