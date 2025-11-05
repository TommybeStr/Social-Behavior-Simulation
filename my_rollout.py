# my_rollout.py
# -*- coding: utf-8 -*-
"""
SelfPlayRollout（最小侵入 + 纯文本JSONL + cond_gt_by_turn 打分 + 广播后同步兜底）
- 只在 tp_rank=0 打日志（logger.info）。
- 解析失败 -> step_reward_raw = -1.0，且不展开子节点。
- 轨迹奖励=均值；最终训练用每步F1作为step_reward（不再均分）。
- 每次生成仅发送 [system, 当前 user]，不携带任何历史 assistant/user（避免长度累积）。
- 适配新版 gold：cond_gt_by_turn = List[List[Dict[parent -> List[str]]]]
- 父键精准匹配：cond_key = 当前 user.user_name 或 "__ROOT__"
- 明细 JSONL 记录 cond_key（不再记录整层父列表）
"""

import sys, os, json, time, logging
from copy import deepcopy
from collections import deque
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.sglang_rollout.sglang_rollout import (
    SGLangRollout,
    broadcast_pyobj,
    logger,
)
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)

# --------------------- Logger：硬编码（stdout, INFO） --------------------- #
try:
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler(stream=sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(h)
except Exception:
    pass

# ====== 纯文本 JSONL（原有总览日志） ====== #
LOG_KEPT_STEP = True
LOG_MODEL_OUTPUT = True
LOG_TO_JSONL = True
ROLL_JSONL_PATH = os.environ.get(
    "ROLL_JSONL_PATH",
    "/home/zss/Social_Behavior_Simulation/rollouts/iter27_rollout.jsonl",
)

_ALLOWED_JSONL_KEYS = {
    "group_id", "trajectory_id", "depth",
    "input_text", "output_text", "parsed_output",
    "step_reward", "step_advantage",
}
_JSONL_READY = False
_JSONL_ERR_ONCE = False

def _log_on(self) -> bool:
    return getattr(self, "_tp_rank", 0) == 0

def _ensure_jsonl_ready():
    global _JSONL_READY, _JSONL_ERR_ONCE
    if _JSONL_READY:
        return True
    try:
        d = os.path.dirname(ROLL_JSONL_PATH)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            try: os.chmod(d, 0o775)
            except Exception: pass
        with open(ROLL_JSONL_PATH, "a", encoding="utf-8") as f:
            f.flush(); os.fsync(f.fileno())
        try: os.chmod(ROLL_JSONL_PATH, 0o664)
        except Exception: pass
        _JSONL_READY = True
        return True
    except Exception as e:
        if not _JSONL_ERR_ONCE:
            _JSONL_ERR_ONCE = True
            try: logger.warning("[rollout][jsonl] cannot open %s: %s", ROLL_JSONL_PATH, e)
            except Exception: pass
        return False

def _jsonl_write_minimal(obj: dict):
    if not LOG_TO_JSONL: return
    if not _ensure_jsonl_ready(): return
    try:
        slim = {k: obj[k] for k in _ALLOWED_JSONL_KEYS if k in obj}
        if not slim: return
        with open(ROLL_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
    except Exception as e:
        if not _JSONL_ERR_ONCE:
            try: logger.warning("[rollout][jsonl] write failed: %s", e)
            except Exception: pass

# ====== ⭐ 新增：明细 JSONL（输入/输出/gold 逐次记录） ====== #
DETAIL_JSONL_PATH = os.environ.get("ROLL_JSONL_DETAIL_PATH", "")
if not DETAIL_JSONL_PATH:
    try:
        base, ext = os.path.splitext(ROLL_JSONL_PATH or "")
        DETAIL_JSONL_PATH = f"{base}_io_gold.jsonl" if base else "/tmp/rollout_io_gold.jsonl"
    except Exception:
        DETAIL_JSONL_PATH = "/tmp/rollout_io_gold.jsonl"

_DETAIL_JSONL_READY = False
_DETAIL_JSONL_ERR_ONCE = False

def _ensure_detail_jsonl_ready():
    global _DETAIL_JSONL_READY, _DETAIL_JSONL_ERR_ONCE
    if _DETAIL_JSONL_READY:
        return True
    try:
        d = os.path.dirname(DETAIL_JSONL_PATH)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            try: os.chmod(d, 0o775)
            except Exception: pass
        with open(DETAIL_JSONL_PATH, "a", encoding="utf-8") as f:
            f.flush(); os.fsync(f.fileno())
        try: os.chmod(DETAIL_JSONL_PATH, 0o664)
        except Exception: pass
        _DETAIL_JSONL_READY = True
        return True
    except Exception as e:
        if not _DETAIL_JSONL_ERR_ONCE:
            _DETAIL_JSONL_ERR_ONCE = True
            try: logger.warning("[rollout][detail-jsonl] cannot open %s: %s", DETAIL_JSONL_PATH, e)
            except Exception: pass
        return False

def _detail_jsonl_write(obj: dict):
    if not _ensure_detail_jsonl_ready(): return
    try:
        with open(DETAIL_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
    except Exception as e:
        if not _DETAIL_JSONL_ERR_ONCE:
            try: logger.warning("[rollout][detail-jsonl] write failed: %s", e)
            except Exception: pass

# --------------------- 常量 & 小工具 --------------------- #
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
_EPS = 1e-6
_VALID_TYPES = {"评论", "转发", "转发微博"}
ROOT_FALLBACK_KEY = "__ROOT__"

def _json_try_load(s: str):
    s = (s or "").strip()
    if not s or s[0] not in "[{": return None
    try: return json.loads(s)
    except Exception: return None

def _parse_and_validate(model_resp: str) -> Tuple[List[Dict[str, str]], str]:
    s = (model_resp or "").strip()
    if not s:
        return [], "parse_fail"

    if s == "[]":  # 新口径：空列表即无交互
        return [], "empty"

    if s == NO_INTERACTION_STR:  # 老兼容
        return [], "empty"

    try:
        parsed = json.loads(s)
    except Exception:
        return [], "parse_fail"

    if not isinstance(parsed, list):
        return [], "parse_fail"

    out = []
    for it in parsed:
        if not isinstance(it, dict):
            return [], "parse_fail"
        u, c, t = it.get("user_name"), it.get("content"), it.get("type")
        if not isinstance(u, str) or not isinstance(c, str) or not isinstance(t, str):
            return [], "parse_fail"
        u, c, t = u.strip(), c.strip(), t.strip()
        if not u or t not in _VALID_TYPES:
            return [], "parse_fail"
        out.append({"user_name": u, "content": c, "type": t})

    if len(out) == 0:
        return [], "empty"

    seen, uniq = set(), []
    for x in out:
        n = x.get("user_name", "")
        if n and n not in seen:
            seen.add(n); uniq.append(x)

    if len(uniq) == 0:
        return [], "empty"
    return uniq, "ok"

def _parse_pred_list(s: str) -> List[Dict[str, str]]:
    preds, status = _parse_and_validate(s)
    return preds if status in ("ok", "empty") else []

def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _to_names(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s == "[]" or s == NO_INTERACTION_STR:
            return []
        parsed = _json_try_load(s)
        if isinstance(parsed, list):
            return _to_names(parsed)
        return [s]

    out: List[str] = []
    if isinstance(x, dict):
        n = (x.get("user_name") or "").strip()
        return [n] if n else []
    if isinstance(x, list):
        for it in x:
            if isinstance(it, str):
                s = it.strip()
                if not s or s == "[]" or s == NO_INTERACTION_STR:
                    continue
                parsed = _json_try_load(s)
                if isinstance(parsed, list):
                    out.extend(_to_names(parsed))
                else:
                    out.append(s)
            elif isinstance(it, dict):
                n = (it.get("user_name") or "").strip()
                if n:
                    out.append(n)
    return [n for n in _dedup_keep_order(out) if n]

def _set_f1(pred_names: List[str], gold_names: List[str]) -> float:
    pset, gset = set([n for n in pred_names if n]), set([n for n in gold_names if n])
    if len(pset) == 0 or len(gset) == 0:   return 0.0
    inter = len(pset & gset)
    prec = inter / (len(pset) + _EPS)
    rec  = inter / (len(gset) + _EPS)
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec + _EPS)

_DROP_KEYS = {"gt_by_turn","gt","gold","labels","answer","evaluation","rewards","reward_model","depth"}

def _sanitize_user_json(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict): return {}
    return {k: v for k, v in d.items() if k not in _DROP_KEYS and not str(k).startswith("_")}

def _render_messages_plain(messages: List[Message]) -> str:
    parts: List[str] = []
    for m in messages or []:
        role = (getattr(m, "role", "") or "").upper()
        content = getattr(m, "content", "")
        parts.append(f"{role}:\n{content}\n")
    return "\n".join(parts)

# ====== list 添加 .tolist()（兼容 VERL 某些路径） ====== #
class _ListWithToList(list):
    def tolist(self): return list(self)

def _deep_wrap_lists_with_tolist(obj):
    if isinstance(obj, list):   return _ListWithToList([_deep_wrap_lists_with_tolist(x) for x in obj])
    if isinstance(obj, tuple):  return tuple(_deep_wrap_lists_with_tolist(x) for x in obj)
    if isinstance(obj, dict):   return {k: _deep_wrap_lists_with_tolist(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object: return _deep_wrap_lists_with_tolist(obj.tolist())
        return obj
    return obj

# ====== 从 prompts 提取 cond_gt_by_turn ====== #
def _extract_cond_gt_from_prompts(prompts: DataProto) -> List[Any]:
    def _dig_one(rm: Any):
        if not isinstance(rm, dict):
            return None
        gt = rm.get("ground_truth")
        if not isinstance(gt, dict):
            gt = rm.get("gt") if isinstance(rm.get("gt"), dict) else {}
        for key in ("cond_gt_by_turn", "cond_gold_by_turn"):
            cgt = gt.get(key)
            if isinstance(cgt, list):
                return cgt
        return None

    ntb = getattr(prompts, "non_tensor_batch", None)
    if isinstance(ntb, dict):
        c = _dig_one(ntb.get("reward_model"))
        if isinstance(c, list): return c
        rms = ntb.get("reward_models")
        if isinstance(rms, list) and rms:
            c = _dig_one(rms[0])
            if isinstance(c, list): return c
        gt = ntb.get("ground_truth")
        if isinstance(gt, dict):
            for key in ("cond_gt_by_turn", "cond_gold_by_turn"):
                c = gt.get(key)
                if isinstance(c, list): return c
        for key in ("cond_gt_by_turn", "cond_gold_by_turn"):
            c = ntb.get(key)
            if isinstance(c, list): return c

    mi = getattr(prompts, "meta_info", None)
    if isinstance(mi, dict):
        c = _dig_one(mi.get("reward_model"))
        if isinstance(c, list): return c

    return []

# ====== 新格式层识别：list[{parent: [children]}] ====== #
def _layer_is_list_of_parent_maps(level: Any) -> bool:
    if not isinstance(level, list):
        return False
    for it in level:
        if not isinstance(it, dict) or len(it) != 1:
            return False
        (k, v), = it.items()
        if not isinstance(k, str) or not isinstance(v, list):
            return False
    return True

# ====== 严格父键口径取 gold：返回（gold_names, cond_key, flags） ====== #
def _strict_gold_for_parent(depth: int, cond_key: str, cond_gt_by_turn: List[Any],
                            fallback_gt_by_turn: Optional[List[Any]] = None
) -> Tuple[List[str], str, Dict[str, bool]]:
    flags = {
        "mode": "none",               # 'list_of_maps' | 'dict' | 'list' | 'none'
        "used_list_level": False,     # 老：整层list当gold
        "parent_missing": False,
        "depth_out_of_range": False,
        "empty_after_parse": False,
        "used_fallback": False,
    }
    ck = (cond_key or "").strip() or ROOT_FALLBACK_KEY
    gold_names: List[str] = []

    # 1) 主 gold
    if isinstance(cond_gt_by_turn, list) and 0 <= depth < len(cond_gt_by_turn):
        level = cond_gt_by_turn[depth]
        if _layer_is_list_of_parent_maps(level):
            flags["mode"] = "list_of_maps"
            found = False
            for m in level:
                (p, ch), = m.items()
                if p == ck:
                    gold_names = _to_names(ch)
                    found = True
                    break
            if not found:
                flags["parent_missing"] = True
        elif isinstance(level, dict):  # 旧数据
            flags["mode"] = "dict"
            if ck in level:
                gold_names = _to_names(level.get(ck, []))
            else:
                flags["parent_missing"] = True
        elif isinstance(level, list):  # 老数据：整层 list 作为 gold
            flags["mode"] = "list"
            flags["used_list_level"] = True
            gold_names = _to_names(level)
        else:
            pass
    else:
        flags["depth_out_of_range"] = True

    # 2) 回退 gold（可选：来自 user.gt_by_turn）
    if not gold_names and isinstance(fallback_gt_by_turn, list) and 0 <= depth < len(fallback_gt_by_turn):
        flags["used_fallback"] = True
        gold_names = _to_names(fallback_gt_by_turn[depth])

    flags["empty_after_parse"] = (len(gold_names) == 0)
    return gold_names, ck, flags

# =================================================== #
# rollout
# =================================================== #
class SelfPlayRollout(SGLangRollout):
    KEEP_PI = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _log_on(self):
            try:
                logger.info(
                    "[rollout][init] tp_rank=%s max_model_len=%s prompt_len=%s resp_len=%s calc_logp=%s",
                    getattr(self, "_tp_rank", None),
                    getattr(self.config, "max_model_len", None),
                    getattr(self.config, "prompt_length", None),
                    getattr(self.config, "response_length", None),
                    bool(getattr(self.config, "calculate_log_probs", False)),
                )
                if _ensure_jsonl_ready():
                    logger.info("[rollout][jsonl] text-only step logs -> %s (exists=%s)",
                                os.path.abspath(ROLL_JSONL_PATH),
                                os.path.exists(ROLL_JSONL_PATH))
                if _ensure_detail_jsonl_ready():
                    logger.info("[rollout][detail-jsonl] io+gold step logs -> %s (exists=%s)",
                                os.path.abspath(DETAIL_JSONL_PATH),
                                os.path.exists(DETAIL_JSONL_PATH))
            except Exception:
                pass

    # ---------- 同步兜底：确保所有 TP rank 在返回前对齐 ---------- #
    def _sync_tp(self):
        try:
            tp_group = self._device_mesh_cpu["tp"].get_group()
        except Exception:
            tp_group = None
        try:
            if dist.is_available() and dist.is_initialized() and tp_group is not None:
                dist.barrier(group=tp_group)
        except Exception:
            try:
                tdev = self._device if hasattr(self, "_device") else "cpu"
                t = torch.ones(1, device=tdev)
                dist.all_reduce(t, group=tp_group)
            except Exception:
                pass

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        return self._req_level_generate_sequences(prompts, **kwargs)

    def _load_trajectory_reward_fn(self) -> Optional[Callable]:
        path = getattr(self.config.multi_turn, "trajectory_reward_fn", None)
        if not path: return None
        try:
            mod, func = path.split(":")
            return getattr(__import__(mod, fromlist=[func]), func)
        except Exception as e:
            if _log_on(self):
                logger.info("Failed to load trajectory_reward_fn '%s': %s", path, e)
            return None

    async def _generate_one_step(self, req: AsyncRolloutRequest, request_sampling_params: dict):
        gen_prompt_ids = req.get_generation_prompt_ids(self.processing_class)
        if len(gen_prompt_ids) + 1 >= self.config.max_model_len:
            if _log_on(self):
                logger.info("[rollout][skip] prompt too long: %d >= max_model_len-1(%d)",
                            len(gen_prompt_ids), self.config.max_model_len - 1)
            return "", FinishReasonTypeEnum.LENGTH

        remaining = self.config.max_model_len - len(gen_prompt_ids) - 1
        max_new = request_sampling_params.get("max_new_tokens", self.config.response_length)
        safe_max_new = max(0, min(max_new, remaining))
        sampling_params = dict(request_sampling_params)
        sampling_params["max_new_tokens"] = safe_max_new

        output = await self._handle_engine_call(req, sampling_params, image_data=None)
        content = output.get("text", "")
        meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
        fr = meta.get("finish_reason", {})
        finish_type = fr.get("type") if isinstance(fr, dict) else None
        finish_reason = FinishReasonTypeEnum.from_str(finish_type or "stop")
        return content, finish_reason

    def _clone_req_with_messages(self, base_req: AsyncRolloutRequest, messages: List[Message], request_id: Optional[str] = None) -> AsyncRolloutRequest:
        return AsyncRolloutRequest(
            batch_data_id=base_req.batch_data_id,
            rollout_offset=base_req.rollout_offset,
            request_id=str(request_id or base_req.request_id),
            state=AsyncRolloutRequestStateEnum.RUNNING,
            messages=list(messages),
            multi_modal_data=base_req.multi_modal_data,
            tool_schemas=None,
            tools_kwargs={},
            interaction_kwargs={},
            input_ids=None,
            attention_mask=None,
            response_ids=[],
            response_attention_mask=[],
            response_position_ids=[],
            response_loss_mask=[],
            reward_scores={},
            max_prompt_len=self.config.prompt_length,
            max_response_len=self.config.response_length,
            max_model_len=self.config.max_model_len,
            use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
            tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
            processing_class=self.processing_class,
        )

    def _build_children_user_jsons_unrestricted(self, parent_user_json_full: Dict[str, Any], model_resp: str) -> List[Dict[str, Any]]:
        """所有合法预测用户名都展开（不与 gold 求交），保持与父相同 potential_interactors。"""
        preds, status = _parse_and_validate(model_resp)
        if status != "ok" or not preds: return []
        parent_pi = parent_user_json_full.get("potential_interactors", [])
        children = []
        for p in preds:
            u = (p.get("user_name") or "").strip()
            if not u:
                continue
            child = {
                "user_name": u,
                "content": p.get("content", ""),
                "depth": int((parent_user_json_full.get("depth") or 0)) + 1,
                "historical_interactors": list(parent_user_json_full.get("historical_interactors", [])),
                "potential_interactors": deepcopy(parent_pi),
            }
            # 继续携带 gold（评估用；输入前会清洗掉）
            if "reward_model" in parent_user_json_full:
                child["reward_model"] = parent_user_json_full["reward_model"]
            children.append(child)
        return children

    # ====== 基于 cond_gt_by_turn 的打分（父键精准） ====== #
    def _score_with_cond_gt(
        self,
        depth: int,
        cond_key: str,
        content: str,
        cond_gt_by_turn: List[Any],
        fallback_gt_by_turn: Optional[List[Any]] = None,
    ) -> Tuple[float, List[str], List[str], str, str]:
        preds, status = _parse_and_validate(content)
        if status == "parse_fail":
            return -1.0, [], [], status, cond_key

        pred_names = [d["user_name"] for d in preds] if preds else []

        gold_names, ck, flags = _strict_gold_for_parent(depth, cond_key, cond_gt_by_turn, fallback_gt_by_turn)
        if _log_on(self) and (flags.get("parent_missing") or flags.get("depth_out_of_range")):
            try:
                logger.warning("[rollout][gold] depth=%d cond_key=%s missing=%s out_of_range=%s used_fallback=%s",
                               depth, ck, flags.get("parent_missing"), flags.get("depth_out_of_range"), flags.get("used_fallback"))
            except Exception:
                pass

        f1 = _set_f1(pred_names, gold_names)
        return float(f1), pred_names, gold_names, status, ck

    @torch.no_grad()
    async def _bfs_expand_and_collect_raw(
        self,
        root_req: AsyncRolloutRequest,
        request_sampling_params: dict,
        max_assistant_turns: int,
        max_children_per_node: Optional[int],
        traj_reward_fn: Optional[Callable],
        trajectory_id: str,
        sidecar_cond_gt: Optional[List[Any]] = None
    ) -> Tuple[List[AsyncRolloutRequest], float, Dict[str, Any]]:
        group_id = str(root_req.request_id)

        # 仅提取根 system
        root_system = None
        for m in root_req.messages:
            if getattr(m, "role", "") == "system":
                root_system = Message(role="system", content=getattr(m, "content", ""))
                break
        if root_system is None:
            root_system = Message(role="system", content="")

        # 根 user 的 full JSON（含 private gold）
        root_user_full = None
        for m in reversed(root_req.messages):
            if m.role == "user":
                root_user_full = _json_try_load(m.content) or {}
                break
        root_user_full = root_user_full or {}
        root_user_full.setdefault("historical_interactors", [])
        root_user_full.setdefault("potential_interactors", [])

        q = deque([root_user_full])

        steps = 0
        step_reqs: List[AsyncRolloutRequest] = []
        traj_user_pred: List[str] = []
        step_rewards_raw: List[float] = []
        pending_jsonl_entries: List[dict] = []
        pending_detail_entries: List[dict] = []

        # 优先用 sidecar 的 cond_gt_by_turn；否则尝试 root_user_full.reward_model
        cond_gt_by_turn: List[Any] = list(sidecar_cond_gt or [])
        if not cond_gt_by_turn:
            try:
                rm = (root_user_full or {}).get("reward_model") or {}
                gt = (rm.get("ground_truth") or {})
                cgt = gt.get("cond_gt_by_turn")
                if isinstance(cgt, list):
                    cond_gt_by_turn = cgt
            except Exception:
                cond_gt_by_turn = []

        # 展开深度上限（相对深度 0/1/2 …）
        DEPTH_LIMIT = 2

        while q and steps < max_assistant_turns:
            cur_user_full = q.popleft()

            # 清洗后送模
            cur_user_clean = _sanitize_user_json(cur_user_full or {})
            run_messages = [root_system, Message(role="user", content=json.dumps(cur_user_clean, ensure_ascii=False))]

            run_req = self._clone_req_with_messages(root_req, run_messages, request_id=group_id)
            content, finish_reason = await self._generate_one_step(run_req, request_sampling_params)
            if not content.strip():
                if _log_on(self): logger.info("[rollout][skip-empty] group=%s traj=%s", group_id, trajectory_id)
                steps += 1; continue
            if finish_reason == FinishReasonTypeEnum.LENGTH:
                if _log_on(self): logger.info("[rollout][skip-length] group=%s traj=%s", group_id, trajectory_id)
                steps += 1; continue

            # kept step
            step_req = self._clone_req_with_messages(root_req, run_messages, request_id=group_id)
            step_req.add_assistant_message(self.processing_class, content)
            step_req.finalize(self.processing_class, {}, FinishReasonTypeEnum.STOP)

            if getattr(self.config, "calculate_log_probs", False):
                await self._maybe_compute_step_logprobs(step_req)

            step_reqs.append(step_req)

            depth = int((cur_user_full or {}).get("depth") or 0)
            fallback_gt = (cur_user_full or {}).get("gt_by_turn") or []
            cond_key = (cur_user_clean.get("user_name") or "").strip() or ROOT_FALLBACK_KEY

            # 评分（严格父键）
            step_f1, pred_names, gold_names, status, cond_key_out = self._score_with_cond_gt(
                depth, cond_key, content, cond_gt_by_turn, fallback_gt
            )

            if status in ("ok", "empty"):
                for u in pred_names:
                    if u: traj_user_pred.append(u)

            step_req.reward_scores = {
                "step_reward_raw": float(step_f1),
                "group_id": group_id,
                "trajectory_id": trajectory_id,
                "depth": depth,
                "finish_reason": finish_reason.name,
                "parse_status": status,
            }

            # —— 控制台日志 —— #
            if _log_on(self) and LOG_KEPT_STEP:
                if LOG_MODEL_OUTPUT:
                    shown = content if len(content) <= 2000 else (content[:1000] + "\n...(omitted)...\n" + content[-800:])
                    logger.info(
                        "[rollout][kept-step] group=%s traj=%s depth=%d status=%s step_F1(raw)=%.4f cond_key=%s gold=%s pred=%s\n===MODEL OUTPUT BEGIN===\n%s\n===MODEL OUTPUT END===",
                        group_id, trajectory_id, depth, status, float(step_f1),
                        cond_key_out,
                        json.dumps(gold_names, ensure_ascii=False),
                        json.dumps(pred_names, ensure_ascii=False),
                        shown,
                    )
                else:
                    logger.info(
                        "[rollout][kept-step] group=%s traj=%s depth=%d status=%s step_F1(raw)=%.4f cond_key=%s gold=%s pred=%s",
                        group_id, trajectory_id, depth, status, float(step_f1),
                        cond_key_out,
                        json.dumps(gold_names, ensure_ascii=False),
                        json.dumps(pred_names, ensure_ascii=False),
                    )

            # —— 原有总览 JSONL —— #
            if _log_on(self) and LOG_TO_JSONL:
                try:
                    prompt_text = _render_messages_plain(run_messages)
                    pending_jsonl_entries.append({
                        "group_id": group_id,
                        "trajectory_id": trajectory_id,
                        "depth": depth,
                        "input_text": prompt_text,
                        "output_text": content,
                        "parsed_output": _parse_pred_list(content),
                        "step_reward": float(step_f1),
                    })
                except Exception:
                    pass

            # ⭐ —— 明细 JSONL（输入/输出/gold/cond_key） —— ⭐
            if _log_on(self):
                try:
                    prompt_text = _render_messages_plain(run_messages)
                    pending_detail_entries.append({
                        "ts": int(time.time() * 1000),
                        "group_id": group_id,
                        "trajectory_id": trajectory_id,
                        "depth": depth,
                        "input_user": cur_user_clean,
                        "input_text": prompt_text,
                        "output_text": content,
                        "gold": gold_names,
                        "cond_key": cond_key_out,
                        "parse_status": status,
                        "finish_reason": finish_reason.name,
                    })
                except Exception:
                    pass

            # 展开子节点（仅 ok 且 depth < DEPTH_LIMIT）——所有预测（含不在 gold 的）都入队
            if status == "ok":
                if "reward_model" not in cur_user_full:
                    cur_user_full["reward_model"] = {"ground_truth": {"cond_gt_by_turn": cond_gt_by_turn}}
                if depth < DEPTH_LIMIT:
                    children_full = self._build_children_user_jsons_unrestricted(cur_user_full, content)
                    for child_full in children_full:
                        child_full["reward_model"] = {"ground_truth": {"cond_gt_by_turn": cond_gt_by_turn}}
                        q.append(child_full)
                else:
                    if _log_on(self):
                        logger.info("[rollout][depth-cap] depth=%d reached limit=%d, stop expanding children",
                                    depth, DEPTH_LIMIT)

            steps += 1

        traj_meta = {
            "trajectory_id": str(trajectory_id),
            "trajectory_pred_users": _dedup_keep_order(traj_user_pred),
        }
        raw_reward_default = float(np.mean([float(r) for r in step_rewards_raw]) if step_rewards_raw else 0.0)
        raw_reward = raw_reward_default  # 可插 trajectory_reward_fn 做自定义聚合

        kept_steps = len(step_reqs)

        # 使用每步F1作为该步的step_reward；轨迹均值仅日志参考
        for s in step_reqs:
            prev = s.reward_scores or {}
            step_f1 = float(prev.get("step_reward_raw", 0.0))
            s.reward_scores = {
                **prev,
                "step_reward": step_f1,
                "trajectory_reward": float(raw_reward),
                "group_id": group_id,
                "trajectory_id": trajectory_id,
            }

        # 写日志
        if _log_on(self) and LOG_TO_JSONL and pending_jsonl_entries:
            for e in pending_jsonl_entries:
                _jsonl_write_minimal(e)
            pending_jsonl_entries.clear()
        if _log_on(self) and pending_detail_entries:
            for e in pending_detail_entries:
                _detail_jsonl_write(e)
            pending_detail_entries.clear()

        if _log_on(self):
            logger.info(
                "[rollout][traj] group=%s traj=%s kept_steps=%d traj_reward=%.4f preds=%s",
                group_id, trajectory_id, kept_steps, float(raw_reward),
                json.dumps(traj_meta.get("trajectory_pred_users", []), ensure_ascii=False),
            )

        return step_reqs, raw_reward, traj_meta

    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # 递归包装 list -> 带 .tolist()
        CHAT_KEYS = {"raw_prompt", "prompt", "messages", "raw_messages"}
        for container in (getattr(prompts, "non_tensor_batch", None), getattr(prompts, "meta_info", None)):
            if isinstance(container, dict):
                for k in list(container.keys()):
                    if k in CHAT_KEYS or isinstance(container[k], list):
                        container[k] = _deep_wrap_lists_with_tolist(container[k])
        if isinstance(prompts.non_tensor_batch, dict):
            prompts.non_tensor_batch = _deep_wrap_lists_with_tolist(prompts.non_tensor_batch)
        if isinstance(prompts.meta_info, dict):
            prompts.meta_info = _deep_wrap_lists_with_tolist(prompts.meta_info)

        # 采样参数
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update({
                "n": 1, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "repetition_penalty": 1.0, "temperature": 0, "top_p": 1,
                "top_k": -1, "ignore_eos": False, "min_new_tokens": 0,
                "max_new_tokens": self.config.response_length,
                "skip_special_tokens": True, "spaces_between_special_tokens": True,
            })
        elif is_validate:
            request_sampling_params.update({"top_k": 50, "top_p": 0.9, "temperature": 0.4, "n": 1})
        request_sampling_params.update(kwargs)

        traj_reward_fn = None  # 如需自定义：self._load_trajectory_reward_fn()
        num_traj = int(getattr(self.config.multi_turn, "num_trajectories", 8))

        # 从 non_tensor_batch / meta_info 提前抽取 cond_gt_by_turn
        sidecar_cond_gt = _extract_cond_gt_from_prompts(prompts)
        if _log_on(self):
            logger.info("[rollout][gold-source] sidecar_cond_gt_len=%s",
                        len(sidecar_cond_gt) if isinstance(sidecar_cond_gt, list) else None)

        # 空批构造器
        def _build_empty_batch():
            P, R = self.config.prompt_length, self.config.response_length
            tgt = tgt_device
            PAD = 0
            tok = getattr(self, "tokenizer", None)
            if tok is not None and getattr(tok, "pad_token_id", None) is not None:
                PAD = int(tok.pad_token_id)
            else:
                eng = getattr(self, "_engine", None)
                tok2 = getattr(eng, "tokenizer", None) if eng is not None else None
                if tok2 is not None and getattr(tok2, "pad_token_id", None) is not None:
                    PAD = int(tok2.pad_token_id)

            prompts_mat   = torch.empty((0, P), dtype=torch.long, device=tgt)
            responses_mat = torch.empty((0, R), dtype=torch.long, device=tgt)
            p_mask_mat    = torch.empty((0, P), dtype=torch.long, device=tgt)
            r_mask_mat    = torch.empty((0, R), dtype=torch.long, device=tgt)
            rloss_mask_mat= torch.empty((0, R), dtype=torch.long, device=tgt)
            rm_scores     = torch.empty((0, R), dtype=torch.float32, device=tgt)

            input_ids      = torch.cat([prompts_mat, responses_mat], dim=-1)
            attention_mask = torch.cat([p_mask_mat, r_mask_mat], dim=-1)
            position_ids   = torch.cumsum(attention_mask, dim=-1) - 1
            position_ids   = position_ids.clamp(min=0).to(dtype=torch.long)

            batch = TensorDict({
                "prompts": prompts_mat, "responses": responses_mat,
                "attention_mask": attention_mask, "input_ids": input_ids,
                "response_mask": r_mask_mat, "response_loss_mask": rloss_mask_mat,
                "loss_mask": rloss_mask_mat, "rm_scores": rm_scores,
                "position_ids": position_ids,
            }, batch_size=0)

            non_tensor_batch = {"uid": np.arange(0, dtype=np.int64)}
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        results = None; err = None
        if getattr(self, "_tp_rank", 0) == 0:
            try:
                try:
                    root_reqs = self._preprocess_prompt_to_async_rollout_requests(prompts, n=1)
                except TypeError:
                    root_reqs = self._preprocess_prompt_to_async_rollout_requests(prompts)

                if _log_on(self):
                    logger.info("[rollout][roots] num_roots=%d num_traj_per_root=%d", len(root_reqs), num_traj)

                import asyncio
                async def _run_all():
                    selected_results = []
                    for req in root_reqs:
                        group_id = str(req.request_id)
                        for t in range(num_traj):
                            per_traj_sampling = dict(request_sampling_params)
                            if "temperature" not in per_traj_sampling or per_traj_sampling["temperature"] == 0:
                                per_traj_sampling["temperature"] = 1.0
                            per_traj_sampling["n"] = 1
                            trajectory_id = f"{group_id}-{t}" if num_traj > 1 else group_id
                            steps, raw_r, meta = await self._bfs_expand_and_collect_raw(
                                root_req=req,
                                request_sampling_params=per_traj_sampling,
                                max_assistant_turns=8,
                                max_children_per_node=getattr(self.config.multi_turn, "max_children_per_node", None),
                                traj_reward_fn=traj_reward_fn,
                                trajectory_id=trajectory_id,
                                sidecar_cond_gt=sidecar_cond_gt,
                            )
                            if len(steps) > 0:
                                compressed = []
                                for s in steps:
                                    compressed.append({
                                        "prompt_ids": (s.prompt_ids.detach().cpu().tolist()
                                            if isinstance(s.prompt_ids, torch.Tensor) else s.prompt_ids),
                                        "response_ids": (s.response_ids.detach().cpu().tolist()
                                            if isinstance(s.response_ids, torch.Tensor) else s.response_ids),
                                        "prompt_attention_mask": (s.prompt_attention_mask.detach().cpu().tolist()
                                            if isinstance(s.prompt_attention_mask, torch.Tensor) else s.prompt_attention_mask),
                                        "response_attention_mask": (s.response_attention_mask.detach().cpu().tolist()
                                            if isinstance(s.response_attention_mask, torch.Tensor) else s.response_attention_mask),
                                        "response_loss_mask": (s.response_loss_mask.detach().cpu().tolist()
                                            if isinstance(s.response_loss_mask, torch.Tensor) else s.response_loss_mask),
                                        "reward_scores": dict(getattr(s, "reward_scores", {}) or {}),
                                        "request_id": getattr(s, "request_id", ""),
                                    })
                                selected_results.append((compressed, meta))
                    return selected_results

                results = asyncio.get_event_loop().run_until_complete(_run_all())

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if _log_on(self):
                    logger.exception("[rollout][rank0-error] %s", err)

        tp_group = self._device_mesh_cpu["tp"].get_group()
        [err, results] = broadcast_pyobj(
            data=[err, results],
            rank=self._rank,
            dist_group=tp_group,
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )

        if err is not None:
            if _log_on(self): logger.info("[rollout][empty-return] rank0 error propagated: %s", err)
            self._sync_tp()
            return _build_empty_batch()

        step_reqs_all: List[Dict[str, Any]] = []
        if results is None:
            if _log_on(self): logger.info("[rollout][diagnose] broadcast results None (rank>0或无数据)")
        else:
            for pair in results:
                if not pair or len(pair) != 2:
                    if _log_on(self): logger.info("[rollout][diagnose] bad pair in results: %s", str(type(pair)))
                    continue
                compressed_steps, _traj_meta = pair
                if compressed_steps: step_reqs_all.extend(compressed_steps)

        if len(step_reqs_all) == 0:
            if _log_on(self): logger.info("[rollout][empty] no kept steps -> return empty batch")
            self._sync_tp()
            return _build_empty_batch()

        try: tp_size = tp_group.size()
        except Exception: tp_size = dist.get_world_size(group=tp_group)
        N = len(step_reqs_all); rem = N % tp_size
        if rem != 0:
            if _log_on(self): logger.info("[rollout][trim] N=%d tp=%d -> trim tail %d", N, tp_size, rem)
            step_reqs_all = step_reqs_all[:-rem]
        if len(step_reqs_all) == 0:
            if _log_on(self): logger.info("[rollout][empty-after-trim] -> return empty batch")
            self._sync_tp()
            return _build_empty_batch()

        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        response_loss_masks = []
        reward_scores = []

        def _to_device_tensor(x, dtype=torch.long):
            if isinstance(x, torch.Tensor):
                t = x.to(dtype=dtype, device=tgt_device)
            else:
                t = torch.as_tensor(x, dtype=dtype, device=tgt_device)
            if t.dim() == 2 and t.size(0) == 1: t = t.squeeze(0)
            return t

        for rec in step_reqs_all:
            p_ids = _to_device_tensor(rec.get("prompt_ids", []), torch.long)
            r_ids = _to_device_tensor(rec.get("response_ids", []), torch.long)
            p_attn = _to_device_tensor(rec.get("prompt_attention_mask", torch.ones_like(p_ids) if p_ids.numel() else torch.tensor([], device=tgt_device)), torch.long)
            r_attn = _to_device_tensor(rec.get("response_attention_mask", torch.ones_like(r_ids) if r_ids.numel() else torch.tensor([], device=tgt_device)), torch.long)
            r_loss = _to_device_tensor(rec.get("response_loss_mask", torch.tensor([], device=tgt_device)), torch.long)
            prompt_ids.append(p_ids); response_ids.append(r_ids)
            prompt_attention_mask.append(p_attn); response_attention_mask.append(r_attn)
            response_loss_masks.append(r_loss)
            reward_scores.append(rec.get("reward_scores", {}))

        PAD = 0
        tok = getattr(self, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            PAD = int(tok.pad_token_id)
        else:
            eng = getattr(self, "_engine", None)
            tok2 = getattr(eng, "tokenizer", None) if eng is not None else None
            if tok2 is not None and getattr(tok2, "pad_token_id", None) is not None:
                PAD = int(tok2.pad_token_id)
            else:
                PAD = int(prompts.meta_info.get("pad_token_id", 0))

        def _pad_stack(seq_list, L, pad_val):
            if len(seq_list) == 0:
                return torch.empty((0, L), dtype=torch.long, device=tgt_device)
            seq_list = [t[: L] for t in seq_list]
            padded = pad_sequence(
                [torch.cat([t, torch.full((L - t.numel(),), pad_val, dtype=t.dtype, device=t.device)]) if t.numel() < L else t for t in seq_list],
                batch_first=True,
            )
            return padded

        P, R = self.config.prompt_length, self.config.response_length
        prompts_mat = _pad_stack(prompt_ids, P, PAD)
        responses_mat = _pad_stack(response_ids, R, PAD)
        p_mask_mat = _pad_stack(prompt_attention_mask, P, 0)
        r_mask_mat = _pad_stack(response_attention_mask, R, 0)
        rloss_mask_mat = _pad_stack(response_loss_masks, R, 0)
        if rloss_mask_mat.numel() == 0 or rloss_mask_mat.shape != r_mask_mat.shape:
            rloss_mask_mat = r_mask_mat

        input_ids = torch.cat([prompts_mat, responses_mat], dim=-1)
        attention_mask = torch.cat([p_mask_mat, r_mask_mat], dim=-1)
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids.clamp(min=0).to(dtype=torch.long)

        response_mask = r_mask_mat

        # 把每步 step_reward 均摊到该步有效 token
        rm_scores = torch.zeros_like(responses_mat, dtype=torch.float32, device=tgt_device)
        step_rewards_arr = torch.tensor(
            [float((rs or {}).get("step_reward", 0.0)) for rs in reward_scores],
            dtype=torch.float32, device=tgt_device,
        )
        for i in range(responses_mat.size(0)):
            valid_idx = (r_mask_mat[i] > 0).nonzero(as_tuple=True)[0]
            L = int(valid_idx.numel())
            if L > 0 and step_rewards_arr[i] != 0.0:
                rm_scores[i, valid_idx] = float(step_rewards_arr[i].item()) / float(L)

        # 过滤无效
        valid = (rloss_mask_mat.sum(dim=-1) > 0)
        if valid.numel() == 0 or int(valid.sum().item()) == 0:
            if _log_on(self):
                logger.info("[rollout][skip-step] loss_mask.sum()==0 -> return empty batch (bs=%d)", responses_mat.size(0))
            self._sync_tp()
            return _build_empty_batch()

        prompts_mat = prompts_mat[valid]; responses_mat = responses_mat[valid]
        p_mask_mat = p_mask_mat[valid]; r_mask_mat = r_mask_mat[valid]
        rloss_mask_mat = rloss_mask_mat[valid]
        input_ids = input_ids[valid]; attention_mask = attention_mask[valid]
        position_ids = position_ids[valid]; response_mask = r_mask_mat
        rm_scores = rm_scores[valid]

        batch = TensorDict({
            "prompts": prompts_mat, "responses": responses_mat,
            "attention_mask": attention_mask, "input_ids": input_ids,
            "response_mask": response_mask, "response_loss_mask": rloss_mask_mat,
            "loss_mask": rloss_mask_mat, "rm_scores": rm_scores,
            "position_ids": position_ids,
        }, batch_size=responses_mat.size(0))

        bs = responses_mat.size(0)
        non_tensor_batch = {"uid": np.arange(bs, dtype=np.int64)}

        if _log_on(self):
            try:
                mean_resp_len = r_mask_mat.sum(dim=-1).float().mean().item() if bs else 0.0
                rm_nonzero_ratio = float((rm_scores.abs().sum(dim=-1) > 0).float().mean().item()) if bs else 0.0
                mean_step_reward = float(
                    torch.tensor([float((rs or {}).get("step_reward", 0.0)) for rs in reward_scores]).mean().item()
                ) if bs else 0.0
            except Exception:
                try:
                    mean_resp_len = r_mask_mat.sum(dim=-1).float().mean().item() if bs else 0.0
                    rm_nonzero_ratio = float((rm_scores.abs().sum(dim=-1) > 0).float().mean().item()) if bs else 0.0
                    mean_step_reward = float(np.mean([float((rs or {}).get("step_reward", 0.0)) for rs in reward_scores])) if bs else 0.0
                except Exception:
                    mean_resp_len = rm_nonzero_ratio = mean_step_reward = 0.0
            try:
                n_zero_step = int(sum(1 for rs in reward_scores if abs(float((rs or {}).get("step_reward", 0.0))) < 1e-12))
                logger.info("[rollout][batch] kept_steps=%d mean_resp_len=%.1f rm_nonzero_ratio=%.2f mean_step_reward=%.4f zero_steps=%d",
                            bs, mean_resp_len, rm_nonzero_ratio, mean_step_reward, n_zero_step)
                logger.info("[rollout][diag] sum(response_mask)=%d, sum(response_loss_mask)=%d",
                            int(r_mask_mat.sum().item()), int(rloss_mask_mat.sum().item()))
            except Exception:
                pass

        # flush sglang cache
        if self._engine is not None and getattr(self, "_tp_rank", 0) == 0:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        # —— 返回前同步兜底 —— #
        self._sync_tp()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
