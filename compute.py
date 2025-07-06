import json
import argparse
from collections import Counter

def main(input_file, output_csv=None):
    """
    统计每条 SFT 样本中“潜在活跃用户”的数量分布。
    输入为 JSONL，每行包含 {id, messages, seq_len}。
    对于每条记录，只统计 depth=0 的 user 消息中的潜在活跃用户长度。
    """
    counts = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # messages 列表
            for msg in rec.get('messages', []):
                # 找到顶层 user 消息 (depth 0)
                if msg.get('role') == 'user' and msg.get('depth') == 0:
                    content = msg.get('content', '')
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # 跳过无法解析的
                        continue
                    # 计算潜在活跃用户数量
                    pa = data.get('潜在活跃用户', [])
                    if isinstance(pa, list):
                        counts.append(len(pa))
                    break  # 每条记录只统计一次 depth=0 的 user

    dist = Counter(counts)

    # 打印分布
    print("潜在活跃用户数量分布 (潜在活跃用户数: 样本数):")
    for num, cnt in sorted(dist.items()):
        print(f"{num}: {cnt}")

    # 输出 CSV
    if output_csv:
        with open(output_csv, 'w', encoding='utf-8') as fout:
            fout.write("potential_active_user_count,sample_count\n")
            for num, cnt in sorted(dist.items()):
                fout.write(f"{num},{cnt}\n")
        print(f"分布结果已保存到 {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='统计 SFT 样本中潜在活跃用户数量的分布'
    )
    parser.add_argument('input', help='输入 SFT JSONL 文件路径')
    parser.add_argument('--output', help='可选的输出 CSV 路径', default=None)
    args = parser.parse_args()
    main(args.input, args.output)
