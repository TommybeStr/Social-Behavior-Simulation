import json
import argparse
import pandas as pd
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/zss/Social_Behavior_Simulation/Qwen2.5-1.5B-Instruct")

def compute_token_length(text):
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def main(jsonl_path, output_csv=None, top_n=None):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                msg_total = 0
                for msg in sample['messages']:
                    msg_total += compute_token_length(msg['content'])
                records.append({
                    'id': sample['id'],
                    'token_len': msg_total
                })

    df = pd.DataFrame(records)
    df.sort_values(by='token_len', ascending=False, inplace=True)

    print(f"共统计 {len(df)} 条样本")
    print(f"最大 token 数: {df['token_len'].max()}")
    print(f"平均 token 数: {df['token_len'].mean():.2f}")
    print(f"最小 token 数: {df['token_len'].min()}")

    if top_n:
        print(f"\nTop {top_n} 最长样本：")
        print(df.head(top_n).to_string(index=False))

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n详细结果已保存至：{output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用 Qwen tokenizer 统计每条 SFT 样本的 token 数')
    parser.add_argument('jsonl_path', help='输入 JSONL 文件路径')
    parser.add_argument('--output_csv', help='可选：输出 CSV 文件路径')
    parser.add_argument('--top_n', type=int, help='可选：显示 token 最长的前 N 条样本')
    args = parser.parse_args()

    main(args.jsonl_path, args.output_csv, args.top_n)
