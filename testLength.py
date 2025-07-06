import json
import argparse
import csv

def main(input_file, output_csv=None):
    """
    对每条 SFT 样本中的每个消息，统计其中“潜在活跃用户”列表的长度（occurrences_count）
    以及列表中独立用户数（unique_count）。
    输入: SFT JSONL，每行包含 {id, messages, seq_len}
    输出: 控制台打印，并可选写入 CSV: record_id, message_idx, role, occurrences_count, unique_count
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = rec.get('id')
            msgs = rec.get('messages', [])
            for idx, msg in enumerate(msgs):
                content = msg.get('content', '')
                # 若 content 为 JSON 字符串
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    continue
                # 检查潜在活跃用户字段
                if '潜在活跃用户' in data:
                    pa_list = data.get('潜在活跃用户', [])
                    if not isinstance(pa_list, list):
                        continue
                    occurrences = len(pa_list)
                    unique_count = len(set(pa_list))
                    results.append({
                        'record_id': record_id,
                        'message_idx': idx,
                        'role': msg.get('role'),
                        'occurrences_count': occurrences,
                        'unique_count': unique_count
                    })
    # 打印
    print("record_id,message_idx,role,occurrences_count,unique_count")
    for r in results:
        print(f"{r['record_id']},{r['message_idx']},{r['role']},{r['occurrences_count']},{r['unique_count']}")
    # 写 CSV
    if output_csv:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['record_id','message_idx','role','occurrences_count','unique_count'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Results written to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计每条消息中潜在活跃用户的次数及人数')
    parser.add_argument('input', help='输入 SFT JSONL 文件路径')
    parser.add_argument('--output', '-o', help='输出 CSV 文件路径', default=None)
    args = parser.parse_args()
    main(args.input, args.output)
