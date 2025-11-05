# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ========= 修改为你的 parquet 路径 =========
PARQUET_PATH = r"/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.15/train.parquet"
EMPTY_STR_LEGACY = "以上用户都不感兴趣，没有发生任何交互"  # 兼容旧数据
VALID_TYPES = {"评论", "转发", "转发微博"}

# --- 将 Arrow 标量转为 Python 原生对象 ---
def _as_py(x):
    return x.as_py() if hasattr(x, "as_py") else x

def normalize_messages(cell):
    """
    把一行的 messages 统一为 Python 的 list[dict]：
      - 旧版: JSON 字符串 -> json.loads(list)
      - Arrow List/Struct 标量 -> 逐项 as_py()
    """
    if cell is None:
        return []
    if isinstance(cell, str):
        try:
            obj = json.loads(cell)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    # Arrow list 或 已是 list
    try:
        it = list(cell)
    except TypeError:
        return []
    out = []
    for m in it:
        m = _as_py(m)
        out.append(m)
    return out

def _try_get_depth_from_content(content_obj):
    """从 content(JSON) 推断 step_depth：优先 depth，否则用 len(historical_interactors)。"""
    step_depth = None
    if isinstance(content_obj, dict):
        if "depth" in content_obj:
            try:
                step_depth = int(content_obj.get("depth"))
                return step_depth
            except Exception:
                pass
        # 用 ancestors 数量做兜底
        hins = content_obj.get("historical_interactors", None)
        if isinstance(hins, list):
            try:
                step_depth = int(len(hins))
                return step_depth
            except Exception:
                pass
    return step_depth

def parse_user_msg(user_msg):
    """
    返回:
      step_depth(int) : 当前 user 节点的深度（0=root 节点）
      reply_layer(int): 业务口径层 = step_depth + 1（1=第一层回复）
      q_user(str|None): 当前 user 的用户名
    兼容：顶层 message.depth / content.depth / content.historical_interactors
    """
    step_depth = user_msg.get("depth", None)
    content = user_msg.get("content", None)
    q_user = None

    # 解析 content
    content_obj = None
    if isinstance(content, str):
        try:
            content_obj = json.loads(content)
        except Exception:
            content_obj = None
    elif isinstance(content, dict):
        content_obj = content

    # 补全 step_depth 与 q_user
    if content_obj is not None:
        if step_depth is None:
            step_depth = _try_get_depth_from_content(content_obj)
        q_user = content_obj.get("user_name", None)

    # 兜底转 int
    try:
        step_depth = int(step_depth) if step_depth is not None else None
    except Exception:
        step_depth = None

    if q_user is not None:
        q_user = str(q_user)

    reply_layer = (step_depth + 1) if step_depth is not None else None
    return step_depth, reply_layer, q_user

def parse_assistant_msg(asst_msg):
    """
    返回:
      is_empty (bool): 空数组/无有效项
      item_count (int): 有效条目数（=评论条目数+转发条目数）
      comments (int): item['type']=='评论' 的条目数
      reposts  (int): item['type'] in {'转发','转发微博'} 的条目数
    定义“有效条目”：dict 且 user_name 非空 且 type 在 VALID_TYPES 中
    """
    content = asst_msg.get("content", None)

    # 解析 content 为 list
    parsed = None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
    elif isinstance(content, list):
        parsed = content

    # 判空（修复版是 []；兼容旧版中文占位）
    if not isinstance(parsed, list) or len(parsed) == 0:
        return True, 0, 0, 0
    if len(parsed) == 1 and isinstance(parsed[0], str) and parsed[0].strip() == EMPTY_STR_LEGACY:
        return True, 0, 0, 0

    comments = 0
    reposts = 0
    valid_items = 0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        uname = (item.get("user_name") or "").strip()
        t = (item.get("type") or "").strip()
        if not uname or t not in VALID_TYPES:
            continue
        if t == "评论":
            comments += 1
        elif t in ("转发", "转发微博"):
            reposts += 1
        valid_items += 1

    if valid_items == 0:
        return True, 0, 0, 0

    # 确保恒等：item_count == comments + reposts
    return False, valid_items, comments, reposts

def iter_turns(messages):
    """
    相邻配对：user -> assistant（构造脚本保证紧邻，且按 BFS 层级输出）
    这里返回业务口径层 reply_layer=step_depth+1，便于直接筛选 depth=1/2。
    """
    i = 0
    n = len(messages)
    while i < n - 1:
        u = messages[i]
        v = messages[i + 1]
        i += 1
        if not (isinstance(u, dict) and isinstance(v, dict)):
            continue
        if u.get("role") != "user":
            continue
        if v.get("role") != "assistant":
            continue

        step_depth, reply_layer, q_user = parse_user_msg(u)
        is_empty, item_count, comments, reposts = parse_assistant_msg(v)

        yield {
            "step_depth": step_depth,       # 0-based
            "reply_layer": reply_layer,     # 1-based（业务口径）
            "q_user": q_user,
            "is_empty": is_empty,
            "item_count": item_count,       # = comments + reposts
            "comments": comments,
            "reposts": reposts,
        }

def main():
    p = Path(PARQUET_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Parquet 文件不存在：{p}")

    df = pd.read_parquet(p)  # 建议 engine='pyarrow'
    if "messages" not in df.columns:
        raise ValueError("未在 parquet 中找到 'messages' 列。")

    # 拆成 turn 级别，同时统计“每组对话的轮次数”
    rows = []
    turns_per_dialog_total = []    # 每棵树的全部 user→assistant 配对数
    turns_per_dialog_d12 = []      # 每棵树的 reply_layer∈{1,2} 的配对数

    for cell in df["messages"]:
        msgs = normalize_messages(cell)
        dialog_pairs = list(iter_turns(msgs))  # 一棵树里所有 user→assistant 步
        rows.extend(dialog_pairs)

        # 轮次数（不区分是否空）——一棵树有多少对 user→assistant
        turns_per_dialog_total.append(len(dialog_pairs))
        turns_per_dialog_d12.append(sum(1 for r in dialog_pairs if r.get("reply_layer") in (1, 2)))

    turns = pd.DataFrame(rows)
    if turns.empty:
        # 快速自检
        sample = normalize_messages(df["messages"].iloc[0]) if len(df) else []
        print(">>> DEBUG sample messages len:", len(sample))
        print(">>> DEBUG first elem type:", type(sample[0]) if sample else None)
        raise RuntimeError("未解析出任何 (user, assistant) 轮，请检查 messages 结构。")

    # 仅统计 业务口径 depth=1/2
    turns = turns[turns["reply_layer"].isin([1, 2])].copy()
    if turns.empty:
        raise RuntimeError("解析到轮次，但未命中 reply_layer ∈ {1,2}。请检查数据或深度写入/推断逻辑。")

    # 只统计 ok/empty 两类（其余脏状态已在 parse 中归并）
    countable = turns.copy()

    # —— 有效交互（按条目计数）：sum(item_count) over ok；无效交互：空数组步的次数
    df_ok = countable[~countable["is_empty"]].copy()
    df_empty = countable[countable["is_empty"]].copy()

    eff = (
        df_ok.groupby("reply_layer")
        .agg(
            effective=("item_count", "sum"),         # 条目级
            comments_total=("comments", "sum"),
            reposts_total=("reposts", "sum"),
        )
        .reset_index()
    )

    inv = (
        df_empty.groupby("reply_layer")
        .size()
        .rename("invalid")
        .reset_index()
    )

    # 合并并补零
    merged = eff.merge(inv, on="reply_layer", how="outer").fillna(0)
    for col in ["effective","comments_total","reposts_total","invalid"]:
        merged[col] = merged[col].astype(int)

    # total 与占比
    merged["total"] = merged["effective"] + merged["invalid"]
    merged["effective_ratio"] = merged["effective"] / merged["total"].replace(0, np.nan)

    # 表 1：交互数
    inter_df = (
        merged[["reply_layer","effective","invalid","total","effective_ratio"]]
        .rename(columns={"reply_layer":"depth"})
        .sort_values("depth")
        .reset_index(drop=True)
    )

    # 表 2：有效交互内的类型分布（与 inter_df.effective 完全对齐）
    types_df = (
        merged[["reply_layer","comments_total","reposts_total"]]
        .rename(columns={"reply_layer":"depth"})
        .sort_values("depth")
        .reset_index(drop=True)
    )

    # ========== 打印结果 ==========
    print("\n=== (1) 有效/无效交互（业务口径 depth=1/2；有效=条目级；无效=空数组步数）===")
    print(inter_df.to_string(index=False))

    print("\n=== (2) 有效交互中的评论/转发条目数（depth=1/2）===")
    print(types_df.to_string(index=False))

    # 新增：每组对话（每棵树）的平均轮次数
    avg_turns_all = float(np.mean(turns_per_dialog_total)) if turns_per_dialog_total else 0.0
    avg_turns_d12 = float(np.mean(turns_per_dialog_d12)) if turns_per_dialog_d12 else 0.0
    print("\n=== (3) 平均每组对话包含的轮次数 ===")
    print(f"• 全部层级（含>2层）：{avg_turns_all:.4f} 轮/组")
    print(f"• 业务口径 depth1/2：{avg_turns_d12:.4f} 轮/组")

    # 额外一致性校验（可选）
    chk = (types_df["comments_total"] + types_df["reposts_total"]).values
    effv = inter_df["effective"].values
    if len(chk) == len(effv) and not np.all(chk == effv):
        print("\n[WARN] 一致性校验失败：comments_total + reposts_total 与 effective 不一致，请检查数据。")

if __name__ == "__main__":
    main()
