# -*- coding: utf-8 -*-
"""
多分类头（按 depth 路由）评估版（适配 head1 为 2 分类）— 精简日志版（改造版）

功能：
- 从 GRPO parquet 中读取样本（含 prompt/messages）
- 对每条样本做 BFS 展开：
  * depth==0 用 cls_head0（3类：0=无互动，1=评论，2=转发）
  * depth>=1 用 cls_head_ge1（2类：0=无互动，1=有互动；type=2 不再出现）
- 每一步：
  * 用分类头对候选池逐个打 type
  * 用完整输入调用一次 LM 生成整串输出（通常为 JSON 数组）
  * 对于分类头判定为 type==1 的候选，第 k 个候选的评论从整串生成里解析出来：
      - 先按 user_name 匹配
      - 再按候选索引 idx 回退
      - 若解析失败则该候选 content 置为空（<|cstart|><|cend|>）
  * 将 type==1 的候选作为下一层 BFS 的 parent 节点，节点内容为解析得到的评论文本
- 只输出两份日志：
  1) jsonl_detail：供 stats_from_logs.py 使用，字段结构保持兼容（output_text 为 list[dict]）
  2) jsonl_io：记录每一步模型的输入文本（chat_text）与「模型真实原始输出字符串」（raw generation text）

注意：
- 本脚本不再在内部做任何 F1 / 图级统计，只负责写日志。
"""

import os, re, json, argparse, time
from collections import deque
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AddedToken

# ===== 常量 =====
PSEP_TOKEN   = "<|psep|>"
PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END   = "\n</POTENTIAL_SPANS>\n"
CSTART_TOKEN = "<|cstart|>"
CEND_TOKEN   = "<|cend|>"
ROOT_FALLBACK_KEY = "__ROOT__"
EPS = 1e-6

_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def extract_username_from_content(s: str) -> str:
    if not isinstance(s, str): return ""
    m = _USERNAME_LINE_RE.search(s)
    return (m.group("name").strip() if m else "")

# ===== 小工具 =====
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _as_py(x):
    return x.as_py() if hasattr(x, "as_py") else x

def normalize_prompt(cell):
    """兼容 prompt/messages 两种列格式，统一还原为 list[dict]"""
    if cell is None:
        return []
    if isinstance(cell, str):
        try:
            obj = json.loads(cell)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    try:
        it = list(cell)
    except TypeError:
        one = _as_py(cell)
        return [one] if isinstance(one, dict) else []
    out = []
    for m in it:
        out.append(_as_py(m))
    return out

def _ensure_special_tokens(tokenizer: AutoTokenizer):
    needed = [PSEP_TOKEN, CSTART_TOKEN, CEND_TOKEN]
    cur = list(tokenizer.special_tokens_map.get("additional_special_tokens", []))
    cur_texts = set([t.content if isinstance(t, AddedToken) else str(t) for t in cur])
    add = [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False)
           for t in needed if t not in cur_texts]
    if add:
        tokenizer.add_special_tokens({"additional_special_tokens": cur + add})
    return tokenizer

# ===== gold 严格对齐（与原逻辑一致）=====
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

def strict_gold_for_parent(depth: int, parent_key: str, cond_gt_by_turn: List[Any]) -> Tuple[List[str], str, Dict[str, bool]]:
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

# ===== F1（单步，用于 step_reward）=====
def set_f1(pred_names: List[str], gold_names: List[str]) -> float:
    pset = set(n for n in pred_names if n)
    gset = set(n for n in gold_names if n)
    if len(gset) == 0: return 0.0
    if len(pset) == 0: return 0.0
    inter = len(pset & gset)
    prec = inter / (len(pset) + EPS)
    rec  = inter / (len(gset) + EPS)
    if (prec + rec) == 0: return 0.0
    return 2 * prec * rec / (prec + rec + EPS)

# ===== POTENTIAL_SPANS 工具 =====
_PSEP_BLOCK_RE = re.compile(r"(<POTENTIAL_SPANS>\s*)(?P<body>.*?)(\s*</POTENTIAL_SPANS>)", re.DOTALL)
_DEPTH_FIELD_RE = re.compile(r'("depth"\s*:\s*)(\d+)')

def strip_psep_block(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    return _PSEP_BLOCK_RE.sub("", text)

def render_psep_block_from_list(pots: List[Dict[str, Any]], depth: int) -> str:
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
        parts.append(json.dumps(blk, ensure_ascii=False, separators=(',', ':')))
        parts.append(PSEP_TOKEN)
    parts.append(PSEP_BLOCK_END)
    return "".join(parts)

def bump_potential_depths(content_text: str, inc: int = 1) -> str:
    if not isinstance(content_text, str) or inc == 0:
        return content_text
    def _bump_in_block(m: re.Match) -> str:
        head, body, tail = m.group(1), m.group("body"), m.group(3)
        def _bump_depth(dm: re.Match) -> str:
            prefix, num = dm.group(1), dm.group(2)
            try: v = max(0, int(num) + inc)
            except Exception: v = num
            return f"{prefix}{v}"
        bumped_body = _DEPTH_FIELD_RE.sub(_bump_depth, body)
        return f"{head}{bumped_body}{tail}"
    return _PSEP_BLOCK_RE.sub(_bump_in_block, content_text)

# ===== 从 reward_model.root_potential 读取候选池 =====
def get_root_potential_objs(user_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    rm = user_json.get("reward_model") or {}
    rp = rm.get("root_potential") or {}
    pots = []
    if isinstance(rp, dict):
        full = rp.get("full")
        if isinstance(full, list) and full:
            for it in full:
                if isinstance(it, dict):
                    name = (it.get("user_name") or "").strip()
                    if name:
                        pots.append({"user_name": name, "interests": it.get("interests") or []})
        elif isinstance(rp.get("user_names"), list):
            for name in rp.get("user_names") or []:
                name = (name or "").strip()
                if name:
                    pots.append({"user_name": name, "interests": []})
    return pots

def get_root_potential_names(user_json: Dict[str, Any]) -> List[str]:
    return [p["user_name"] for p in get_root_potential_objs(user_json)]

# ===== 文本→token 映射（分类 span 用）=====
def find_user_blob_in_chat(chat_text: str, user_blob: str) -> int:
    return chat_text.find(user_blob)

def iter_span_char_ranges_from_user_blob(user_blob: str) -> List[Tuple[int, int]]:
    spans = []
    token = PSEP_TOKEN
    L = len(token)
    pos = 0
    marks = []
    while True:
        i = user_blob.find(token, pos)
        if i == -1: break
        marks.append(i)
        pos = i + L
    for a, b in zip(marks[0::2], marks[1::2]):
        spans.append((a, b + L))
    return spans

def map_char_span_to_token_span(offsets, c_l: int, c_r: int) -> Tuple[int, int]:
    t_l, t_r = -1, -1
    for i, (s, e) in enumerate(offsets):
        if e > c_l:
            t_l = i; break
    if t_l == -1: return -1, -1
    for j in range(t_l, len(offsets)):
        s, e = offsets[j]
        if s >= c_r:
            t_r = j; break
    if t_r == -1: t_r = len(offsets)
    if t_l >= t_r: return -1, -1
    return t_l, t_r

# ===== cstart/cend 工具 & interests 查询 =====
def strip_c_markers(text: str) -> str:
    """从 '<|cstart|>...<|cend|>' 中抽出中间内容，若没有标记则原样返回。"""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    s = text
    if CSTART_TOKEN in s:
        s = s.split(CSTART_TOKEN, 1)[1]
    if CEND_TOKEN in s:
        s = s.split(CEND_TOKEN, 1)[0]
    return s.strip()

def find_user_interests_from_root_potential(root_user_json: Dict[str, Any], uname: str) -> List[Any]:
    uname = (uname or "").strip()
    if not uname:
        return []
    pots = get_root_potential_objs(root_user_json)
    for p in pots:
        if (p.get("user_name") or "").strip() == uname:
            return p.get("interests") or []
    return []

# ===== 多头：按 depth 路由（适配 3类/2类）=====
class DepthwiseHeads(nn.Module):
    def __init__(self, head0: nn.Module, head_ge1: Optional[nn.Module] = None):
        super().__init__()
        self.head0 = head0                  # 3类
        self.head_ge1 = head_ge1 if head_ge1 is not None else head0  # 2类（或兜底）
    def select(self, depth: int) -> nn.Module:
        return self.head0 if int(depth) == 0 else self.head_ge1

def _new_linear_head(hidden: int, num_labels: int, device, dtype):
    head = nn.Linear(hidden * 2, num_labels, device=device, dtype=dtype)
    with torch.no_grad():
        if head.bias is not None:
            head.bias.zero_()
    return head

def _load_head_state_into(head: nn.Linear, state: Dict[str, torch.Tensor]) -> None:
    sd = {}
    if "weight" in state:
        sd["weight"] = state["weight"]
    elif "cls_head.weight" in state:
        sd["weight"] = state["cls_head.weight"]
    else:
        for k,v in state.items():
            if k.endswith("weight") and v.ndim==2 and v.shape==head.weight.shape:
                sd["weight"] = v
    if head.bias is not None:
        if "bias" in state:
            sd["bias"] = state["bias"]
        elif "cls_head.bias" in state:
            sd["bias"] = state["cls_head.bias"]
        else:
            for k,v in state.items():
                if k.endswith("bias") and v.ndim==1 and v.shape==head.bias.shape:
                    sd["bias"] = v
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[cls_head][warn] missing={list(missing)} unexpected={list(unexpected)}")

def load_cls_head_from_file(path: str, hidden: int, num_labels: int, device, dtype) -> nn.Linear:
    head = _new_linear_head(hidden, num_labels, device, dtype)
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load cls head from {path}: {e}")
    if isinstance(obj, nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            state = obj
    else:
        raise RuntimeError(f"Unsupported cls head format in {path}: type={type(obj)}")
    state = {k: (v.to(dtype=dtype) if torch.is_tensor(v) else v) for k,v in state.items()}
    _load_head_state_into(head, state)
    with torch.no_grad():
        w_norm = float(head.weight.norm().item())
        b_norm = float(head.bias.norm().item()) if head.bias is not None else 0.0
    print(f"[cls_head] loaded from {path}  w_norm={w_norm:.6f} b_norm={b_norm:.6f}  dev={head.weight.device} dtype={head.weight.dtype}")
    return head

# ===== 重建输入（为编码 + chat_text）=====
def rebuild_user_view_and_order_with_root_potential(user_json: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    user_view = dict(user_json)
    cur_depth = int(user_view.get("depth", user_view.get("_step_depth", 0)) or 0)

    pots = get_root_potential_objs(user_json)
    content_raw = user_view.get("content", "") or ""
    base_content = strip_psep_block(content_raw)
    if pots:
        psep_block = render_psep_block_from_list(pots, depth=cur_depth)
        new_content = (base_content.rstrip() + psep_block)
    else:
        new_content = content_raw

    user_view["content"] = new_content
    user_view.pop("reward_model", None)
    user_view.pop("ground_truth", None)
    user_view.pop("gold", None)

    # 按 root_potential 顺序确定 cand_order（和 POTENTIAL_SPANS 顺序一致）
    cand_order = [p["user_name"] for p in pots] if pots else \
        [m.group(1).strip() for m in re.finditer(r'\{"user_name"\s*:\s*"(.*?)"', new_content)]
    return user_view, cand_order

# ===== 生成结果解析：从整串 LM 输出中恢复每个 type==1 候选的评论 =====
def _extract_json_array(text: str) -> Optional[str]:
    """从生成文本中粗暴截取第一个 JSON 数组片段，用于 json.loads."""
    if not isinstance(text, str):
        return None
    l = text.find("[")
    r = text.rfind("]")
    if l == -1 or r == -1 or r <= l:
        return None
    return text[l:r+1]

def parse_generated_comments(
    gen_text: str,
    cand_order: List[str],
    pred_types: List[int],
) -> Dict[int, str]:
    """
    解析模型一次性生成的整串输出，提取每个候选的评论文本。
    返回: {候选索引idx -> 纯评论文本（不含 <|cstart|><|cend|>）}
    仅对 pred_types[idx] == 1 的人尝试解析。
    """
    idx_to_comment: Dict[int, str] = {}
    if not isinstance(gen_text, str) or not gen_text.strip():
        return idx_to_comment

    parsed_obj = None

    # 先尝试截出 JSON 数组部分再 loads
    json_seg = _extract_json_array(gen_text)
    if json_seg is not None:
        try:
            parsed_obj = json.loads(json_seg)
        except Exception:
            parsed_obj = None

    # 如果失败，尝试直接整体 loads（兼容模型只输出纯 JSON 的情况）
    if parsed_obj is None:
        try:
            parsed_obj = json.loads(gen_text)
        except Exception:
            return idx_to_comment

    if not isinstance(parsed_obj, list):
        return idx_to_comment

    gen_list = parsed_obj

    # 先按 index / user_name 建两个字典
    by_index: Dict[int, str] = {}
    by_name: Dict[str, str] = {}

    for i, item in enumerate(gen_list):
        if not isinstance(item, dict):
            continue
        uname = (item.get("user_name") or "").strip()
        content_raw = item.get("content") or ""
        comment = strip_c_markers(str(content_raw))
        # 如果 comment 为空字符串，就当无效
        if not comment:
            continue
        by_index[i] = comment
        if uname:
            by_name[uname] = comment

    # 针对每个 type==1 的候选，按姓名优先，其次 index 回退
    for idx, t in enumerate(pred_types):
        if int(t) != 1:
            continue
        uname_target = cand_order[idx] if idx < len(cand_order) else f"__CAND_{idx}__"
        uname_target = (uname_target or "").strip()

        comment = None
        if uname_target and uname_target in by_name:
            comment = by_name[uname_target]
        elif idx in by_index:
            comment = by_index[idx]

        if comment:
            idx_to_comment[idx] = comment

    return idx_to_comment

# ===== 单步：一次编码做分类 + 一次整体生成 + 解析 =====
@torch.no_grad()
def classify_and_decode_one_step(
    model,
    tokenizer,
    device,
    system_prompt: str,
    user_json: Dict[str, Any],
    *,
    heads: DepthwiseHeads,
    decode: bool = True,
    gen_max_new_tokens: int = 48,
    gen_temperature: float = 0.7,
    gen_top_p: float = 0.9,
    max_input_tokens: int = 4096,
    decode_batch_size: int = 8,  # 保留参数以兼容 CLI，但当前逻辑未使用
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    返回:
      - preds: list[dict]，每个元素为 {"user_name", "content", "type"}，供 BFS 和 stats_from_logs 使用
      - chat_text: 本步真实输入给模型的字符串（包含 system + user JSON）
      - gen_text: LM 的原始生成文本（为一次整体 generate 的结果，可能是 JSON / 带噪声的 JSON）
    """

    # ===== 1) 构造统一的 user_view / cand_order / chat_text =====
    user_view, cand_order = rebuild_user_view_and_order_with_root_potential(user_json)
    cur_depth = int(user_json.get("depth", user_json.get("_step_depth", 0)) or 0)
    head = heads.select(cur_depth)

    user_blob = json.dumps(user_view, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": user_blob},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ===== 2) 编码 + forward，拿 hidden 做分类 =====
    enc = tokenizer(
        [chat_text],
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]   # (seq_len, hidden)
    attn_mask   = enc["attention_mask"][0]

    base = find_user_blob_in_chat(chat_text, user_blob)
    if base < 0:
        base = 0

    char_spans_local = iter_span_char_ranges_from_user_blob(user_blob)
    if not char_spans_local:
        # 没有候选 span，直接返回空结果
        return [], chat_text, ""

    user_l, user_r = base, base + len(user_blob)
    t_gl, t_gr = map_char_span_to_token_span(offsets, user_l, user_r)
    if t_gl != -1 and t_gr - t_gl > 1:
        global_vec = last_hidden[t_gl:t_gr, :].mean(dim=0)
    else:
        global_vec = (last_hidden * attn_mask.unsqueeze(-1)).sum(dim=0) / (attn_mask.sum() + 1e-6)

    pred_types: List[int] = []
    for idx, (cl, cr) in enumerate(char_spans_local):
        C_l, C_r = base + cl, base + cr
        t_l, t_r = map_char_span_to_token_span(offsets, C_l, C_r)
        if t_l == -1 or t_r == -1 or t_r - t_l <= 1:
            span_mean = torch.zeros_like(global_vec)
        else:
            span_mean = last_hidden[t_l:t_r, :].mean(dim=0)
        feat = torch.cat([span_mean, global_vec], dim=-1).to(
            device=head.weight.device,
            dtype=head.weight.dtype,
        )
        logits = head(feat).float()
        probs = torch.softmax(logits, dim=-1)
        t = int(torch.argmax(probs, dim=-1).item())
        pred_types.append(t)

    # ===== 3) 若无需生成或没有 type==1，则 content 全部置空，gen_text 为空 =====
    if (not decode) or (all(t != 1 for t in pred_types)):
        results: List[Dict[str, Any]] = []
        for i in range(len(char_spans_local)):
            uname = cand_order[i] if i < len(cand_order) else f"__CAND_{i}__"
            results.append({
                "user_name": uname,
                "content": f"{CSTART_TOKEN}{CEND_TOKEN}",
                "type": int(pred_types[i]),
            })
        return results, chat_text, ""

    # ===== 4) 用完整 chat_text 调一次 generate，拿到 LM 原始输出 =====
    gen_enc = tokenizer(
        [chat_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    gen_enc = {k: v.to(device) for k, v in gen_enc.items()}

    gen_out = model.generate(
        **gen_enc,
        do_sample=True,
        top_p=float(gen_top_p),
        temperature=float(gen_temperature),
        max_new_tokens=int(gen_max_new_tokens),
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        pad_token_id=(
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.pad_token_id
        ),
    )
    inp_len = gen_enc["input_ids"].shape[1]
    new_ids = gen_out[0, inp_len:]
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ===== 5) 从整串生成中解析每个 type==1 候选的评论 =====
    idx_to_comment = parse_generated_comments(gen_text, cand_order, pred_types)

    # ===== 6) 融合：type 以分类头为准，评论来自解析，解析失败则视为空 =====
    results: List[Dict[str, Any]] = []
    num_cands = len(char_spans_local)
    for i in range(num_cands):
        uname = cand_order[i] if i < len(cand_order) else f"__CAND_{i}__"
        t_val = int(pred_types[i])

        if t_val == 1 and i in idx_to_comment:
            cmt = idx_to_comment[i]
            content = f"{CSTART_TOKEN}{cmt}{CEND_TOKEN}"
        else:
            # 包括 type != 1 或 解析失败（没在 idx_to_comment 里）
            content = f"{CSTART_TOKEN}{CEND_TOKEN}"

        results.append({
            "user_name": uname,
            "content": content,
            "type": t_val,
        })

    return results, chat_text, gen_text

# ===== 主评估：只写日志（detail + io）=====
def evaluate_parquet_file(
    parquet_path: str,
    model_path: str,
    *,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    max_samples: int = 100,
    max_turns: int = 20,
    temperature: float = 0.7,
    depth_limit: int = 1,
    jsonl_detail_path: str = "rollout_io_gold.jsonl",
    jsonl_io_path: str = "rollout_model_io.jsonl",
    gen_max_new_tokens: int = 48,
    gen_top_p: float = 0.9,
    no_decode: bool = False,
    max_input_tokens: int = 4096,
    decode_batch_size: int = 8,
    cls_head0_path: Optional[str] = None,
    cls_head_ge1_path: Optional[str] = None,
):
    print(f"[INFO] load base model from: {model_path}")
    if jsonl_detail_path:
        ensure_dir(jsonl_detail_path)
        print(f"[LOG] detail -> {jsonl_detail_path}")
    if jsonl_io_path:
        ensure_dir(jsonl_io_path)
        print(f"[LOG] io     -> {jsonl_io_path}")

    # tokenizer & model
    tknzr_path = tokenizer_path if tokenizer_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(tknzr_path, trust_remote_code=True, use_fast=True)
    tokenizer = _ensure_special_tokens(tokenizer)
    try:
        tokenizer.truncation_side = "left"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype="auto", trust_remote_code=True
    ).eval()
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    ref_param = next(model.parameters())
    dev, dtype = ref_param.device, ref_param.dtype
    hidden = model.config.hidden_size

    # 3类 head0 + 2类 head_ge1
    if cls_head0_path is None and cls_head_ge1_path is None:
        print("[cls_head] no external heads provided; init random heads (3cls + 2cls).")
        head0 = _new_linear_head(hidden, 3, dev, dtype)   # 3类
        head_ge1 = _new_linear_head(hidden, 2, dev, dtype) # 2类
        with torch.no_grad():
            if head0.bias is not None:
                head0.bias.zero_()
            if head_ge1.bias is not None:
                head_ge1.bias.zero_()
        heads = DepthwiseHeads(head0, head_ge1)
        model.cls_head0 = head0  # 兼容 probe
    else:
        head0 = load_cls_head_from_file(cls_head0_path, hidden, 3, dev, dtype) if cls_head0_path else _new_linear_head(hidden, 3, dev, dtype)
        head_ge1 = load_cls_head_from_file(cls_head_ge1_path, hidden, 2, dev, dtype) if cls_head_ge1_path else _new_linear_head(hidden, 2, dev, dtype)
        heads = DepthwiseHeads(head0, head_ge1)
        model.cls_head0 = head0  # 兼容 probe

    # 读取数据
    df = pd.read_parquet(parquet_path)
    if ("prompt" not in df.columns) and ("messages" not in df.columns):
        raise ValueError("数据集中缺少 'prompt' 或 'messages' 列。")

    n_rows = min(int(len(df)), int(max_samples))
    print(f"[INFO] evaluating rows: {n_rows}")

    for ridx, row in tqdm(list(df.iloc[:n_rows].iterrows()), total=n_rows, desc="Evaluating"):
        try:
            raw = row.get("prompt", None)
            if raw is None and "messages" in row:
                raw = row["messages"]
            prompt = normalize_prompt(raw)
            if not isinstance(prompt, list) or len(prompt) < 2:
                print(f"[warn] row {ridx}: invalid prompt")
                continue

            # system prompt
            sys_prompt = ""
            for m in prompt:
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_prompt = m.get("content", "") or ""
                    break

            # 找到首个 user
            first_user_idx = None
            for i, m in enumerate(prompt):
                if isinstance(m, dict) and m.get("role") == "user":
                    first_user_idx = i
                    break
            if first_user_idx is None:
                print(f"[warn] row {ridx}: no user")
                continue

            user_str = prompt[first_user_idx].get("content", "")
            try:
                root_user_json = json.loads(user_str) if isinstance(user_str, str) else None
            except Exception:
                root_user_json = None
            if not isinstance(root_user_json, dict):
                print(f"[warn] row {ridx}: user content not json")
                continue

            rm = root_user_json.get("reward_model") or {}
            gt = rm.get("ground_truth") or {}
            cond_gt_by_turn = gt.get("cond_gt_by_turn") or []
            if not isinstance(cond_gt_by_turn, list):
                cond_gt_by_turn = []

            # 取 record_id
            sft_ci = row.get("sft_chunk_info")
            if isinstance(sft_ci, str):
                try:
                    sft_ci = json.loads(sft_ci)
                except Exception:
                    sft_ci = {}
            record_id = ""
            if isinstance(sft_ci, dict):
                record_id = str(sft_ci.get("record_id") or "")
            if not record_id:
                record_id = f"row{getattr(row, 'name', 'NA')}"

            # BFS 队列
            q = deque()
            root = dict(root_user_json)
            root.setdefault("historical_interactors", [])
            if not root.get("user_name"):
                root["user_name"] = extract_username_from_content(root.get("content", "") or "") or ROOT_FALLBACK_KEY

            start_depth = int(root.get("depth", root.get("_step_depth", 0)) or 0)
            root["_step_depth"] = start_depth
            root["depth"] = start_depth
            q.append(root)

            steps = 0
            traj_id = f"{record_id}-traj0"

            while q and steps < max_turns:
                parent_full = q.popleft()
                step_depth = int(parent_full.get("_step_depth", 0))

                # ====== 调用分类 + 生成（一次整体生成 + 解析） ======
                preds, chat_text, raw_gen_text = classify_and_decode_one_step(
                    model, tokenizer, device=next(model.parameters()).device,
                    system_prompt=sys_prompt, user_json=parent_full,
                    heads=heads,
                    decode=(not no_decode),
                    gen_max_new_tokens=int(gen_max_new_tokens),
                    gen_temperature=float(temperature),
                    gen_top_p=float(gen_top_p),
                    max_input_tokens=int(max_input_tokens),
                    decode_batch_size=int(decode_batch_size),
                )

                # gold 对齐
                cond_key = (parent_full.get("user_name") or "").strip() or ROOT_FALLBACK_KEY
                gold_names, ck_out, flags = strict_gold_for_parent(step_depth, cond_key, cond_gt_by_turn)
                gold_set = set(n for n in gold_names if n)

                # 当前步预测的正例用户名（以分类头 type 为准）
                pred_pos_all = {p["user_name"] for p in preds if int(p.get("type", 0)) in (1, 2) and p.get("user_name")}
                step_f1 = set_f1(list(pred_pos_all), list(gold_set))

                # ====== detail 日志（供 stats_from_logs 使用，结构保持原样）=====
                if jsonl_detail_path:
                    detail = {
                        "ts": int(time.time() * 1000),
                        "group_id": record_id,
                        "trajectory_id": traj_id,
                        "depth": step_depth,
                        "report_depth": step_depth + 1,
                        "cond_key": ck_out,
                        "input_user": parent_full,
                        "output_text": preds,                     # list[dict]，stats_from_logs 会解析
                        "gold": sorted(list(gold_set)),           # list[str]
                        "gold_level_mode": flags.get("mode"),
                        "gold_used_list_level": bool(flags.get("used_list_level")),
                        "gold_parent_missing": bool(flags.get("parent_missing")),
                        "gold_depth_out_of_range": bool(flags.get("depth_out_of_range")),
                        "gold_empty_after_parse": bool(flags.get("empty_after_parse")),
                        "used_fallback": bool(flags.get("fallback_to_empty")),
                        "step_reward": float(step_f1),            # F1
                        "finish_reason": "stop",
                    }
                    with open(jsonl_detail_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(detail, ensure_ascii=False) + "\n")

                # ====== io 日志：记录真实输入字符串 + 模型原始输出字符串 ======
                if jsonl_io_path:
                    io_rec = {
                        "ts": int(time.time() * 1000),
                        "group_id": record_id,
                        "trajectory_id": traj_id,
                        "depth": step_depth,
                        "report_depth": step_depth + 1,
                        "input_text": chat_text,           # 真实喂给 tokenizer 的文本
                        "output_text": raw_gen_text,       # 模型一次 generate 的原始输出字符串
                    }
                    with open(jsonl_io_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(io_rec, ensure_ascii=False) + "\n")

                # ====== BFS 展开：用 type==1 的候选，节点内容为该候选的评论 ======
                if step_depth < depth_limit:
                    parent_name = (parent_full.get("user_name") or "").strip()
                    next_depth = step_depth + 1

                    # 下一层候选池仍然用 root 的全集
                    next_pots = get_root_potential_objs(root_user_json)

                    # 累积历史互动链
                    prev_hist = parent_full.get("historical_interactors") or []
                    prev_names = []
                    for h in prev_hist:
                        if isinstance(h, dict):
                            n = (h.get("user_name") or "").strip()
                            if n:
                                prev_names.append(n)
                        elif isinstance(h, str):
                            n = h.strip()
                            if n:
                                prev_names.append(n)
                    if parent_name:
                        prev_names.append(parent_name)

                    for it in preds:
                        if int(it.get("type", 0)) != 1:
                            continue
                        uname = (it.get("user_name") or "").strip()
                        if not uname:
                            continue

                        # 1) 抽出该候选的评论文本（只取 <|cstart|>...<|cend|> 中间部分）
                        raw_c = it.get("content") or ""
                        comment_text = strip_c_markers(raw_c)

                        # 2) 从 root_potential 找 interests
                        u_interests = find_user_interests_from_root_potential(root_user_json, uname)

                        # 3) 按 SFT 模板重建该节点的 base 内容（不含 POTENTIAL_SPANS）
                        user_plain_prefix = (
                            "username: " + uname + "\n"
                            "content:\n" + comment_text + "\n"
                            "userinterest: " + json.dumps(u_interests, ensure_ascii=False) + "\n"
                            "historicalinteractors: " + json.dumps(prev_names, ensure_ascii=False) + "\n"
                            "potentialspan:"
                        )

                        # 4) 追加下一层 POTENTIAL_SPANS
                        if next_pots:
                            psep_block_next = render_psep_block_from_list(next_pots, depth=next_depth)
                            next_content_for_child = user_plain_prefix + psep_block_next
                        else:
                            next_content_for_child = user_plain_prefix

                        child = {
                            "user_name": uname,
                            "content": next_content_for_child,
                            "historical_interactors": [{"user_name": parent_name}] if parent_name else [],
                            "_step_depth": next_depth,
                            "depth": next_depth,
                            "reward_model": {
                                "ground_truth": {"cond_gt_by_turn": cond_gt_by_turn},
                                "root_potential": (root_user_json.get("reward_model", {}) or {}).get("root_potential"),
                            },
                        }
                        q.append(child)

                steps += 1

        except torch.cuda.OutOfMemoryError as e:
            print(f"[!] row {ridx} failed: CUDA OOM ({str(e)[:120]}...)")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[!] row {ridx} failed: {e}")
            continue

    print("[DONE] rollout finished; all stats 请用 stats_from_logs.py 读 jsonl_detail 计算。")

# ===== CLI =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO parquet with BFS rollout (depth-wise heads) and write logs for stats_from_logs."
    )
    parser.add_argument("--data", type=str, required=True, help="输入 GRPO Parquet 文件（含 prompt/messages）")
    parser.add_argument("--model", type=str, required=True, help="基础模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer 路径（未提供则回退到 --model）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")
    parser.add_argument("--max_samples", type=int, default=10000, help="最大样本条数")
    parser.add_argument("--max_turns", type=int, default=20, help="每个对话的最大展开步数")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度（仅对评论生效）")
    parser.add_argument("--depth_limit", type=int, default=1, help="展开深度上限（0基）")

    parser.add_argument("--jsonl_detail", type=str, default="rollout_io_gold.jsonl",
                        help="明细 JSONL 输出路径（供 stats_from_logs.py 使用）")
    parser.add_argument("--jsonl_io", type=str, default="rollout_model_io.jsonl",
                        help="模型输入/输出文本 JSONL 路径")

    parser.add_argument("--gen_max_new_tokens", type=int, default=48, help="评论生成的最大 token 数")
    parser.add_argument("--gen_top_p", type=float, default=0.9, help="nucleus sampling 的 p")
    parser.add_argument("--no_decode", action="store_true", help="仅分类不生成评论（用于纯分类评估）")
    parser.add_argument("--decode_batch_size", type=int, default=8, help="生成阶段的小批量大小（当前逻辑未使用）")

    parser.add_argument("--max_input_tokens", type=int, default=8192, help="编码/生成输入的最大长度（左截断）")

    parser.add_argument("--cls_head0", dest="cls_head0", type=str, default=None,
                        help="depth==0 使用的分类头 .pt（3类）")
    parser.add_argument("--cls_head_ge1", dest="cls_head_ge1", type=str, default=None,
                        help="depth>=1 使用的分类头 .pt（2类）")

    args = parser.parse_args()

    evaluate_parquet_file(
        parquet_path=args.data,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        depth_limit=args.depth_limit,
        jsonl_detail_path=args.jsonl_detail,
        jsonl_io_path=args.jsonl_io,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_top_p=args.gen_top_p,
        no_decode=args.no_decode,
        max_input_tokens=args.max_input_tokens,
        decode_batch_size=args.decode_batch_size,
        cls_head0_path=args.cls_head0,
        cls_head_ge1_path=args.cls_head_ge1,
    )
