import pandas as pd
import argparse

def split_parquet(input_path, train_path, val_path, val_ratio, seed):
    # 读取 Parquet
    df = pd.read_parquet(input_path)
    # 随机打乱
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # 切分
    split_idx = int(len(df_shuffled) * (1 - val_ratio))
    train_df = df_shuffled.iloc[:split_idx]
    val_df   = df_shuffled.iloc[split_idx:]
    # 保存为 Parquet
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"总样本数: {len(df_shuffled)}, 训练: {len(train_df)}, 验证: {len(val_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机切分 Parquet 数据集为训练集和验证集")
    parser.add_argument('--input',     required=True, help="输入 Parquet 文件路径")
    parser.add_argument('--train_output', required=True, help="输出训练集 Parquet 路径")
    parser.add_argument('--val_output',   required=True, help="输出验证集 Parquet 路径")
    parser.add_argument('--val_ratio',   type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument('--seed',        type=int,   default=42,  help="随机种子，默认 42")
    args = parser.parse_args()

    split_parquet(
        args.input, args.train_output, args.val_output,
        args.val_ratio, args.seed
    )