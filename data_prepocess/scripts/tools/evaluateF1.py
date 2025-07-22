import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from tqdm import tqdm

device = "cuda"

# 模型加载
tokenizer = AutoTokenizer.from_pretrained("/home/zss/Social_Behavior_Simulation/checkpoints/default/global_step_2079")
model = AutoModelForCausalLM.from_pretrained("/home/zss/Social_Behavior_Simulation/checkpoints/default/global_step_2079", device_map="auto").eval()


def extract_predicted_users(response_text):
    """
    从模型生成内容中提取预测的 user_name 列表
    """
    try:
        # 尝试反转义（例如 \"）
        if isinstance(response_text, str):
            response_text = response_text.strip()
            if response_text.startswith('"[{') and response_text.endswith('}]"'):
                response_text = json.loads(response_text)

        if isinstance(response_text, str):
            parsed = json.loads(response_text)
        elif isinstance(response_text, list):
            parsed = response_text
        else:
            return []

        if isinstance(parsed, list):
            return [entry.get("user_name", "").strip() for entry in parsed if "user_name" in entry]
    except Exception as e:
        print(f"[!] 模型输出解析失败: {e}")
    return []


def extract_ground_truth_users(assistant_content):
    try:
        parsed = json.loads(assistant_content)
        return [entry.get("user_name", "").strip() for entry in parsed if "user_name" in entry]
    except:
        return []


def extract_potential_users(user_content):
    try:
        parsed = json.loads(user_content)
        return [entry["user_name"].strip() for entry in parsed.get("potential_interactors", [])]
    except:
        return []


def evaluate_sample(user_content, assistant_content, potential_users):
    messages = [
        {
            "role": "system",
            "content": "你是一个社交媒体互动预测专家，能够根据输入博文的具体内容，预测该条博文的互动情况。你现在收到的输入包括以下字段：- user_name: 原始发布者用户名 - user_interests: 原始发布者兴趣 - content: 博文正文 - depth: 博文在网络中的深度 - historical_interactors: 历史活跃用户 - potential_interactors: 潜在活跃用户列表（你只能从中选人进行预测）你必须严格按照以下格式输出，不允许包含任何解释性内容，也不要展示推理过程：[{\\\"user_name\\\": \\\"用户名（来自potential_interactors）\\\", \\\"content\\\": \\\"预测的评论内容\\\", \\\"type\\\": \\\"评论 或 转发\\\"}, ...] 注意事项：1. 你必须且只能从 potential_interactors 中选择用户填入输出结果；2. type 字段只能为 \\\"评论\\\" 或 \\\"转发\\\"；3. 不允许添加任何说明、理由、分析等内容；4. 输出必须是且只含一个合法的JSON数组结构。\'"},
        {"role": "user", "content": user_content}
    ]

    # 构造输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    pred_users = extract_predicted_users(response)
    true_users = extract_ground_truth_users(assistant_content)
    all_users = potential_users

    y_true = [1 if user in true_users else 0 for user in all_users]
    y_pred = [1 if user in pred_users else 0 for user in all_users]

    return precision_score(y_true, y_pred, zero_division=0), \
           recall_score(y_true, y_pred, zero_division=0), \
           f1_score(y_true, y_pred, zero_division=0)


def evaluate_parquet_file(parquet_path, max_samples=100):
    import numpy as np

    df = pd.read_parquet(parquet_path)

    def safe_string(x):
        # 如果本身是字符串，直接返回
        if isinstance(x, str):
            return x
        # 如果是 numpy 单元素数组
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                return str(x[0])
            else:
                # 多元素情况，尝试拼接（仅调试用）
                return str(x.tolist())
        # 如果是 list-like
        elif isinstance(x, (list, tuple)):
            if len(x) == 1:
                return str(x[0])
            else:
                return str(x)
        # 否则强制转字符串
        return str(x)

    df["messages"] = df["messages"].apply(safe_string)

    precisions, recalls, f1s = [], [], []

    for idx, row in tqdm(df.iterrows(), total=min(len(df), max_samples), desc="Evaluating"):
        if idx >= max_samples:
            break
        try:
            import ast
            messages = ast.literal_eval(row["messages"])
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
            if not user_msg or not assistant_msg:
                continue

            user_content = user_msg["content"]
            assistant_content = assistant_msg["content"]
            potential_users = extract_potential_users(user_content)

            p, r, f1 = evaluate_sample(user_content, assistant_content, potential_users)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        except Exception as e:
            print(f"[!] 样本 {idx} 解析失败: {e}")
            continue

    print(f"\n✅ 共评估样本数: {len(f1s)}")
    print(f"Precision: {sum(precisions)/len(precisions):.4f}")
    print(f"Recall:    {sum(recalls)/len(recalls):.4f}")
    print(f"F1 Score:  {sum(f1s)/len(f1s):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="输入的parquet路径")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    evaluate_parquet_file(args.data, args.max_samples)
