import json

def _maybe_float(x, *keys):
    if isinstance(x, dict):
        for k in keys:
            if k in x:
                try:
                    return float(x[k])
                except Exception:
                    pass
    return None

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    优先返回你在 rollout 阶段已经算好的分数：
      1) extra_info.reward_scores.rewards / trajectory_reward / step_reward
      2) 兼容 kwargs.reward_scores 同名字段
    若都没有，则安全返回 0.0（不会中断训练；真正的 token-level 奖励来自 batch['rm_scores']）。
    """
    # 解析 extra_info
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            extra_info = {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    # 1) 从 extra_info.reward_scores 读取
    rs = extra_info.get("reward_scores", {})
    v = _maybe_float(rs, "rewards", "trajectory_reward", "step_reward")
    if v is not None:
        return v

    # 2) 兼容 extra_info 顶层 或 kwargs.reward_scores
    v = _maybe_float(extra_info, "rewards", "trajectory_reward", "step_reward")
    if v is not None:
        return v
    v = _maybe_float(kwargs.get("reward_scores", {}), "rewards", "trajectory_reward", "step_reward")
    if v is not None:
        return v

    # 3) 兜底：返回 0.0（不影响你已注入的 batch['rm_scores']）
    return 0.0