import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from pathlib import Path

def read_token_counts(csv_file):
    """
    读取CSV文件中的token计数信息
    """
    df = pd.read_csv(csv_file)
    # 假设CSV列名为 '行号' 和 '总 Token 数'
    # 如果列名不同，请根据实际情况调整
    token_counts = {}
    for _, row in df.iterrows():
        line_num = row['行号']
        token_count = row['总 Token 数']
        token_counts[line_num] = token_count
    
    return token_counts

def filter_jsonl_by_tokens(jsonl_file, token_counts, min_tokens=None, max_tokens=None):
    """
    根据token数量过滤JSONL文件
    """
    filtered_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num in token_counts:
                token_count = token_counts[line_num]
                
                # 检查是否在token范围内
                if min_tokens is not None and token_count < min_tokens:
                    continue
                if max_tokens is not None and token_count > max_tokens:
                    continue
                
                # 解析JSON并添加token信息
                try:
                    json_obj = json.loads(line.strip())
                    json_obj['token_count'] = token_count
                    json_obj['line_number'] = line_num
                    filtered_data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"警告: 第{line_num}行JSON格式错误，跳过")
                    continue
    
    return filtered_data

def save_to_parquet(data, output_file):
    """
    将数据保存为Parquet格式
    """
    if not data:
        print("没有数据需要保存")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存为Parquet
    df.to_parquet(output_file, index=False)
    print(f"数据已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='过滤JSONL文件并转换为Parquet格式')
    parser.add_argument('--jsonl', required=True, help='JSONL文件路径')
    parser.add_argument('--csv', required=True, help='CSV文件路径（包含token计数）')
    parser.add_argument('--output', required=True, help='输出Parquet文件路径')
    parser.add_argument('--min-tokens', type=int, help='最小token数（可选）')
    parser.add_argument('--max-tokens', type=int, help='最大token数（可选）')
    parser.add_argument('--auto-range', action='store_true', help='自动使用25%-75%分位数作为范围')
    
    args = parser.parse_args()
    
    # 读取token计数
    print("读取token计数文件...")
    token_counts = read_token_counts(args.csv)
    
    # 如果启用自动范围，计算分位数
    if args.auto_range:
        token_values = list(token_counts.values())
        q25 = pd.Series(token_values).quantile(0.25)
        q75 = pd.Series(token_values).quantile(0.75)
        min_tokens = int(q25)
        max_tokens = int(q75)
        print(f"自动设置token范围: {min_tokens} - {max_tokens}")
    else:
        min_tokens = args.min_tokens
        max_tokens = args.max_tokens
    
    # 显示统计信息
    token_values = list(token_counts.values())
    print(f"Token统计信息:")
    print(f"  总行数: {len(token_values)}")
    print(f"  最小值: {min(token_values)}")
    print(f"  最大值: {max(token_values)}")
    print(f"  平均值: {sum(token_values)/len(token_values):.2f}")
    print(f"  中位数: {sorted(token_values)[len(token_values)//2]}")
    
    if min_tokens or max_tokens:
        print(f"过滤条件:")
        if min_tokens:
            print(f"  最小token数: {min_tokens}")
        if max_tokens:
            print(f"  最大token数: {max_tokens}")
    
    # 过滤JSONL文件
    print("过滤JSONL文件...")
    filtered_data = filter_jsonl_by_tokens(args.jsonl, token_counts, min_tokens, max_tokens)
    
    print(f"过滤后保留 {len(filtered_data)} 条记录")
    
    # 保存为Parquet
    print("保存为Parquet格式...")
    save_to_parquet(filtered_data, args.output)
    
    print("完成!")

if __name__ == "__main__":
    main()