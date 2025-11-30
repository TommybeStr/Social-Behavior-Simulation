# env_bfs.py
# -*- coding: utf-8 -*-
"""
BFS 环境（不依赖 tokenizer / LM 生成）
-----------------------------------

职责：
- 只负责「树结构 + gold 对齐 + F1 reward + BFS 展开」。
- 不管 tokenizer、不管 generate，只吃分类结果 (pred_types)，产出 step reward。
- 逻辑尽量与 evaluate_parquet_file 中的 BFS 展开保持一致。

使用方式（伪代码）：

    from env_bfs import BFSEnv

    env = BFSEnv(root_user_json, depth_limit=1, max_turns=20)
    env.reset()

    while not env.done:
        parents = env.build_step_inputs()      # List[Dict]，每个就是一个 parent_full
        if not parents:
            break

        step_preds = []
        for parent in parents:
            # 你在这里调用模型：
            # cand_order, logits, _ = model.classify_step(parent, system_prompt)
            # actions, logps = sample_actions_from_logits(logits)
            step_preds.append({
                "parent_full": parent,
                "candidate_names": cand_order,       # List[str]
                "pred_types": actions.tolist(),      # List[int]，长度与 candidate_names 一致
            })

        rewards, done = env.step(step_preds)
        # rewards: List[float]，与 step_preds 顺序一一对应（每个 parent 一个 step F1）

    # env.history 里会记录整条轨迹的详细 step 日志（方便 debug）
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, List, Tuple, Optional

# ===== 常量 =====
ROOT_FALLBACK_KEY = "__ROOT__"
EPS = 1e-6

# 和 SFT/评估脚本一致的 username 提取正则
import re

_USERNAME_LINE_RE = re.compile(
    r"^\s*username:\s*(?P<name>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PSEP_TOKEN = "<|psep|>"
PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END = "\n</POTENTIAL_SPANS>\n"


# ======================================================================
# 小工具函数：与评估脚本保持一致
# ======================================================================

def extract_username_from_content(s: str) -> str:
    """从 user.content 中的 'username: xxx' 行提取用户名。"""
    if not isinstance(s, str):
        return ""
    m = _USERNAME_LINE_RE.search(s)
    return (m.group("name").strip() if m else "")


def _as_py(x):
    return x.as_py() if hasattr(x, "as_py") else x


# ===== gold 严格对齐（与评估脚本一致）=====

def _layer_is_list_of_parent_maps(level: Any) -> bool:
    """
    判断 cond_gt_by_turn[depth] 这一层是否为
    [ {parent: [children...]}, {parent2: [...]}, ... ] 的结构。
    """
    if not isinstance(level, list):
        return False
    for it in level:
        if not isinstance(it, dict) or len(it) != 1:
            return False
        (k, v), = it.items()
        if not isinstance(k, str) or not isinstance(v, list):
            return False
    return True


def strict_gold_for_parent(
    depth: int,
    parent_key: str,
    cond_gt_by_turn: List[Any],
) -> Tuple[List[str], str, Dict[str, bool]]:
    """
    与评估脚本中的 strict_gold_for_parent 一致：
    - 根据 depth & parent_key，从 cond_gt_by_turn 取出该 parent 的 gold children 名单。
    - 返回: (gold_names, 标准化后的 parent_key, flags)
    """
    flags = {
        "mode": "none",
        "used_list_level": False,
        "parent_missing": False,
        "depth_out_of_range": False,
        "empty_after_parse": False,
        "fallback_to_empty": False,
    }
    cond_key: str = (parent_key or "").strip() or ROOT_FALLBACK_KEY
    gold_names: List[str] = []

    if not (isinstance(cond_gt_by_turn, list) and 0 <= depth < len(cond_gt_by_turn)):
        flags["depth_out_of_range"] = True
        flags["fallback_to_empty"] = True
        return gold_names, cond_key, flags

    level = cond_gt_by_turn[depth]
    if _layer_is_list_of_parent_maps(level):
        flags["mode"] = "list_of_maps"
        found = False
        for m in level:
            (p, ch), = m.items()
            if p == cond_key:
                gold_names = [n for n in ch if isinstance(n, str) and n.strip()]
                found = True
                break
        if not found:
            flags["parent_missing"] = True
            flags["fallback_to_empty"] = True
        else:
            flags["empty_after_parse"] = (len(gold_names) == 0)
    else:
        # 兼容旧的 list[dict]，但不强制 parent_key 匹配
        if isinstance(level, list):
            flags["mode"] = "list"
            flags["used_list_level"] = True
            for it in level:
                if isinstance(it, dict) and len(it) == 1:
                    (p, ch), = it.items()
                    if p == cond_key:
                        gold_names = [n for n in ch if isinstance(n, str) and n.strip()]
                        break
        else:
            flags["fallback_to_empty"] = True

    return gold_names, cond_key, flags


# ===== F1（单步）=====

def set_f1(pred_names: List[str], gold_names: List[str]) -> float:
    """
    单步 F1（以用户名集合为单位）：
    - pred_names / gold_names 都是用户名列表。
    - gold 为空时，F1 定义为 0（与评估脚本一致）。
    """
    pset = set(n for n in pred_names if n)
    gset = set(n for n in gold_names if n)
    if len(gset) == 0:
        return 0.0
    if len(pset) == 0:
        return 0.0
    inter = len(pset & gset)
    prec = inter / (len(pset) + EPS)
    rec = inter / (len(gset) + EPS)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + EPS)


# ===== root_potential 相关 =====

def get_root_potential_objs(user_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 reward_model.root_potential 中取出完整候选池：
    - 优先使用 root_potential.full（list[dict]）
    - 退化为 root_potential.user_names（只含名字）
    """
    rm = user_json.get("reward_model") or {}
    rp = rm.get("root_potential") or {}
    pots: List[Dict[str, Any]] = []
    if isinstance(rp, dict):
        full = rp.get("full")
        if isinstance(full, list) and full:
            for it in full:
                if isinstance(it, dict):
                    name = (it.get("user_name") or "").strip()
                    if name:
                        pots.append({
                            "user_name": name,
                            "interests": it.get("interests") or [],
                            "depth": int(it.get("depth", 0)) if "depth" in it else 0,
                        })
        elif isinstance(rp.get("user_names"), list):
            for name in rp.get("user_names") or []:
                name = (name or "").strip()
                if name:
                    pots.append({"user_name": name, "interests": [], "depth": 0})
    return pots


def find_user_interests_from_root_potential(
    root_user_json: Dict[str, Any],
    uname: str,
) -> List[Any]:
    """
    从 root_potential 中找某个 user_name 对应的 interests。
    """
    uname = (uname or "").strip()
    if not uname:
        return []
    pots = get_root_potential_objs(root_user_json)
    for p in pots:
        if (p.get("user_name") or "").strip() == uname:
            return p.get("interests") or []
    return []


def render_psep_block_from_list(pots: List[Dict[str, Any]], depth: int) -> str:
    """
    按 POTENTIAL_SPANS 约定把候选池渲染为文本块：
    <POTENTIAL_SPANS>
    <|psep|>{"user_name":..., "interests":[...], "depth": depth}<|psep|>...
    </POTENTIAL_SPANS>
    """
    parts = [PSEP_BLOCK_START]
    dval = int(depth)
    for p in (pots or []):
        blk = {
            "user_name": (p.get("user_name") or "").strip(),
            "interests": p.get("interests") or [],
            "depth": dval,
        }
        if not blk["user_name"]:
            continue
        parts.append(PSEP_TOKEN)
        parts.append(json.dumps(blk, ensure_ascii=False, separators=(",", ":")))
        parts.append(PSEP_TOKEN)
    parts.append(PSEP_BLOCK_END)
    return "".join(parts)


# ==================================================================