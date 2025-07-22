import json
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        # 模型生成的预测用户（JSON array）
        pred_users = [
            entry.get("user_name", "").strip()
            for entry in json.loads(solution_str)
            if isinstance(entry, dict) and "user_name" in entry
        ]
    except:
        pred_users = []

    try:
        true_users = [
            entry.get("user_name", "").strip()
            for entry in json.loads(ground_truth)
            if isinstance(entry, dict) and "user_name" in entry
        ]
    except:
        true_users = []

    try:
        prompt = extra_info.get("prompt", {})
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        potential_users = [
            u.get("user_name", "").strip()
            for u in prompt.get("potential_interactors", [])
        ]
    except:
        potential_users = []

    # 构造标签
    y_true = [1 if user in true_users else 0 for user in potential_users]
    y_pred = [1 if user in pred_users else 0 for user in potential_users]

    return f1_score(y_true, y_pred, zero_division=0)
