import json
import pandas as pd
from tqdm import tqdm
import argparse


def sft_to_grpo(sft_parquet_path, grpo_parquet_path, data_source="social_f1"):
    df = pd.read_parquet(sft_parquet_path)

    grpo_samples = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        try:
            # 解析 messages 列
            messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

            # 提取角色信息
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

            if not user_msg or not assistant_msg:
                continue

            user_content = user_msg["content"]
            assistant_content = assistant_msg["content"]

            # 尝试解析 user_content 为 dict 以提取 depth 和 potential_interactors
            try:
                parsed_prompt = json.loads(user_content)
            except Exception as e:
                print(f"[!] user content JSON 解析失败: {e}")
                continue

            grpo_sample = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": user_content}],
                "ability": "social_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": assistant_content
                },
                "extra_info": {
                    "prompt": user_content,
                    "depth": parsed_prompt.get("depth", None),
                    "chunk_id": row.get("id", f"sample_{idx}")
                }
            }

            grpo_samples.append(grpo_sample)

        except Exception as e:
            print(f"[!] 样本 {idx} 转换失败: {e}")
            continue

    grpo_df = pd.DataFrame(grpo_samples)
    grpo_df.to_parquet(grpo_parquet_path, index=False)
    print(f"\n✅ 共转换样本数: {len(grpo_samples)}")
    print(f"输出路径: {grpo_parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SFT-style data to GRPO-compatible format for verl")
    parser.add_argument("--sft_parquet", required=True, help="输入的 SFT 数据文件（.parquet）路径")
    parser.add_argument("--grpo_parquet", required=True, help="输出的 GRPO 数据文件（.parquet）路径")
    parser.add_argument("--data_source", default="social_f1", help="数据源标签，用于匹配 reward 函数")
    args = parser.parse_args()

    sft_to_grpo(args.sft_parquet, args.grpo_parquet, args.data_source)
