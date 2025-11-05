# -*- coding: utf-8 -*-
"""
多分类头（按 depth 路由）评估版：
- depth==0 使用 cls_head0，depth>=1 使用 cls_head_ge1（若未提供，则复用 head0）
- 使用新版 GRPO 数据的 reward_model.root_potential 渲染 POTENTIAL_SPANS（随 BFS 深度递增）
- 单次编码分类 + batched 生成，仅对 type==1 生成
- 绝不把 gold 喂给模型（仅日志/评分）
"""

import os, re, json, argparse, time, glob
from collections import defaultdict, deque, Counter
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

# ===== gold 严格对齐 =====
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

# ===== 指标 =====
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

def edge_counts_from_events(events, directed=True) -> Counter:
    c = Counter()
    for u, p, *_ in events:
        if not u or not p: continue
        if directed:
            c[(u, p)] += 1
        else:
            a, b = sorted([u, p])
            c[(a, b)] += 1
    return c

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

# ===== 文本→token 映射 =====
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

# ===== 在线探针（可选）=====
@torch.no_grad()
def probe_cls_head_effect(model, tokenizer, device) -> bool:
    toy_user_full = {
        "user_name": "alice",
        "content": 'username: root\n' + PSEP_TOKEN + '{"user_name":"bob","interests":[],"depth":0}' + PSEP_TOKEN,
        "historical_interactors": []
    }
    user_view = dict(toy_user_full)
    for k in ("reward_model","ground_truth","gold"):
        user_view.pop(k, None)

    messages = [{"role":"system","content":""},{"role":"user","content":json.dumps(user_view, ensure_ascii=False)}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer([chat_text], return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k:v.to(device) for k,v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]
    attn = enc["attention_mask"][0]

    user_blob = json.dumps(user_view, ensure_ascii=False)
    base = find_user_blob_in_chat(chat_text, user_blob)
    if base < 0: return False
    spans = iter_span_char_ranges_from_user_blob(user_blob)
    if not spans: return False
    cl, cr = spans[0]
    tl, tr = map_char_span_to_token_span(offsets, base+cl, base+cr)
    if tl < 0 or tr - tl <= 1: return False

    ul, ur = base, base + len(user_blob)
    tgl, tgr = map_char_span_to_token_span(offsets, ul, ur)
    if tgl != -1 and tgr - tgl > 1:
        global_vec = last_hidden[tgl:tgr, :].mean(dim=0)
    else:
        global_vec = (last_hidden * attn.unsqueeze(-1)).sum(dim=0) / (attn.sum() + 1e-6)

    span_mean = last_hidden[tl:tr, :].mean(dim=0)
    feat = torch.cat([span_mean, global_vec], dim=-1)
    feat = feat.to(device=model.cls_head.weight.device, dtype=model.cls_head.weight.dtype)
    logits_1 = model.cls_head(feat).float().softmax(dim=-1)
    w_bak = model.cls_head.weight.data.clone()
    b_bak = model.cls_head.bias.data.clone() if model.cls_head.bias is not None else None
    perm = torch.randperm(model.cls_head.weight.numel(), device=model.cls_head.weight.device)
    model.cls_head.weight.data = model.cls_head.weight.view(-1)[perm].view_as(model.cls_head.weight)
    if model.cls_head.bias is not None:
        model.cls_head.bias.data = torch.flip(model.cls_head.bias.data, dims=[0])
    logits_2 = model.cls_head(feat).float().softmax(dim=-1)
    model.cls_head.weight.data.copy_(w_bak)
    if b_bak is not None: model.cls_head.bias.data.copy_(b_bak)
    diff = torch.norm(logits_1 - logits_2, p=2).item()
    changed = (int(torch.argmax(logits_1)) != int(torch.argmax(logits_2)))
    print(f"[probe] cls_head softmax L2 diff={diff:.4f}, argmax_changed={changed}")
    return bool(diff > 0.15 or changed)

# ===== 多头：按 depth 路由 =====
class DepthwiseHeads(nn.Module):
    def __init__(self, head0: nn.Module, head_ge1: Optional[nn.Module] = None):
        super().__init__()
        self.head0 = head0
        self.head_ge1 = head_ge1 if head_ge1 is not None else head0
    def select(self, depth: int) -> nn.Module:
        return self.head0 if int(depth) == 0 else self.head_ge1

def _new_linear_head(hidden: int, num_labels: int, device, dtype):
    head = nn.Linear(hidden * 2, num_labels, device=device, dtype=dtype)
    with torch.no_grad():
        if head.bias is not None:
            head.bias.zero_()
    return head

def _load_head_state_into(head: nn.Linear, state: Dict[str, torch.Tensor]) -> None:
    # 兼容多种键名
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
    # 可能是 Module / state_dict / dict(weight,bias) / 包含 state_dict
    if isinstance(obj, nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            state = obj
    else:
        raise RuntimeError(f"Unsupported cls head format in {path}: type={type(obj)}")
    # 转 dtype 后装载
    state = {k: (v.to(dtype=dtype) if torch.is_tensor(v) else v) for k,v in state.items()}
    _load_head_state_into(head, state)
    with torch.no_grad():
        w_norm = float(head.weight.norm().item())
        b_norm = float(head.bias.norm().item()) if head.bias is not None else 0.0
    print(f"[cls_head] loaded from {path}  w_norm={w_norm:.6f} b_norm={b_norm:.6f}  dev={head.weight.device} dtype={head.weight.dtype}")
    return head

# ===== 基于 root_potential 重建输入 =====
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

    cand_order = [p["user_name"] for p in pots] if pots else \
        [m.group(1).strip() for m in re.finditer(r'\{"user_name"\s*:\s*"(.*?)"', new_content)]
    return user_view, cand_order

# ===== 单步：一次编码做分类 + batched 生成 =====
@torch.no_grad()
def classify_and_decode_one_step(
    model,
    tokenizer,
    device,
    system_prompt: str,
    user_json: Dict[str, Any],
    *,
    heads: DepthwiseHeads,            # <<< 新增：多头
    decode: bool = True,
    gen_max_new_tokens: int = 48,
    gen_temperature: float = 0.7,
    gen_top_p: float = 0.9,
    max_input_tokens: int = 4096,
    decode_batch_size: int = 8,
    debug: bool = False,
) -> List[Dict[str, Any]]:

    user_view, cand_order = rebuild_user_view_and_order_with_root_potential(user_json)
    cur_depth = int(user_json.get("depth", user_json.get("_step_depth", 0)) or 0)
    head = heads.select(cur_depth)

    # ===== 1) 编码 =====
    user_blob = json.dumps(user_view, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": user_blob},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer([chat_text], return_tensors="pt", return_offsets_mapping=True,
                    padding=True, truncation=True, max_length=max_input_tokens)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]   # [T, H]
    attn_mask   = enc["attention_mask"][0]   # [T]

    # ===== 2) span 定位 + 分类 =====
    base = find_user_blob_in_chat(chat_text, user_blob)
    if base < 0: base = 0

    char_spans_local = iter_span_char_ranges_from_user_blob(user_blob)
    if not char_spans_local:
        return []

    user_l, user_r = base, base + len(user_blob)
    t_gl, t_gr = map_char_span_to_token_span(offsets, user_l, user_r)
    if t_gl != -1 and t_gr - t_gl > 1:
        global_vec = last_hidden[t_gl:t_gr, :].mean(dim=0)
    else:
        global_vec = (last_hidden * attn_mask.unsqueeze(-1)).sum(dim=0) / (attn_mask.sum() + 1e-6)

    pred_types = []
    valid_cnt = 0
    for idx, (cl, cr) in enumerate(char_spans_local):
        C_l, C_r = base + cl, base + cr
        t_l, t_r = map_char_span_to_token_span(offsets, C_l, C_r)
        if t_l == -1 or t_r == -1 or t_r - t_l <= 1:
            span_mean = torch.zeros_like(global_vec); valid = 0
        else:
            span_mean = last_hidden[t_l:t_r, :].mean(dim=0); valid = 1; valid_cnt += 1
        feat = torch.cat([span_mean, global_vec], dim=-1).to(device=head.weight.device, dtype=head.weight.dtype)
        logits = head(feat).float()
        probs = torch.softmax(logits, dim=-1)
        t = int(torch.argmax(probs, dim=-1).item())
        if debug:
            p0, p1, p2 = [round(x,4) for x in probs.tolist()]
            print(f"[DEBUG-CLS][depth={cur_depth}] cand#{idx} valid={valid} probs={[p0,p1,p2]} -> type={t}")
        pred_types.append(t)
    if debug:
        print(f"[DEBUG-CLS] valid_spans={valid_cnt}/{len(char_spans_local)} (depth={cur_depth})")

    # ===== 3) 无需生成 =====
    if (not decode) or (all(t != 1 for t in pred_types)):
        results = []
        for i in range(len(char_spans_local)):
            uname = cand_order[i] if i < len(cand_order) else f"__CAND_{i}__"
            results.append({"user_name": uname, "content": f"{CSTART_TOKEN}{CEND_TOKEN}", "type": int(pred_types[i])})
        return results

    # ===== 4) batched 生成（仅 type==1）=====
    comment_idxs = [i for i, t in enumerate(pred_types) if t == 1]
    start_prompts_suffix = [(cand_order[i] if i < len(cand_order) else f"__CAND_{i}__") + "\n" for i in comment_idxs]
    full_prompts = [chat_text + s for s in start_prompts_suffix]

    gen_texts: List[str] = [""] * len(comment_idxs)
    for b0 in range(0, len(full_prompts), int(decode_batch_size)):
        sub = full_prompts[b0:b0+int(decode_batch_size)]
        tok = tokenizer(sub, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            gen_out = model.generate(
                **tok,
                do_sample=True,
                top_p=float(gen_top_p),
                temperature=float(gen_temperature),
                max_new_tokens=int(gen_max_new_tokens),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
            )
        inp_len = tok["input_ids"].shape[1]
        for j in range(gen_out.size(0)):
            new_ids = gen_out[j, inp_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            gen_texts[b0 + j] = text

    fill_map = {comment_idxs[j]: gen_texts[j] for j in range(len(comment_idxs))}
    results = []
    for i in range(len(char_spans_local)):
        uname = cand_order[i] if i < len(cand_order) else f"__CAND_{i}__"
        if i in fill_map and fill_map[i]:
            content = f"{CSTART_TOKEN}{fill_map[i]}{CEND_TOKEN}"
        else:
            content = f"{CSTART_TOKEN}{CEND_TOKEN}"
        results.append({"user_name": uname, "content": content, "type": int(pred_types[i])})
    return results

# ===== 主评估（BFS，按 depth 递增并选头）=====
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
    jsonl_overview_path: Optional[str] = "rollout_overview.jsonl",
    jsonl_detail_path: Optional[str] = "rollout_io_gold.jsonl",
    report_path: Optional[str] = None,
    graph_directed: bool = True,
    gen_max_new_tokens: int = 48,
    gen_top_p: float = 0.9,
    no_decode: bool = False,
    debug_cls: bool = False,
    assert_cls_loaded: bool = False,
    probe_cls: bool = False,
    max_input_tokens: int = 4096,
    decode_batch_size: int = 8,
    cls_head0_path: Optional[str] = None,        # <<< 新增
    cls_head_ge1_path: Optional[str] = None,     # <<< 新增
):
    print(f"[INFO] load base model from: {model_path}")
    if jsonl_overview_path: ensure_dir(jsonl_overview_path); print(f"[LOG] overview -> {jsonl_overview_path}")
    if jsonl_detail_path:  ensure_dir(jsonl_detail_path);  print(f"[LOG] detail   -> {jsonl_detail_path}")
    if report_path:        ensure_dir(report_path);        print(f"[LOG] report   -> {report_path}")

    # tokenizer & model
    tknzr_path = tokenizer_path if tokenizer_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(tknzr_path, trust_remote_code=True, use_fast=True)
    tokenizer = _ensure_special_tokens(tokenizer)
    try: tokenizer.truncation_side = "left"
    except Exception: pass

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype="auto", trust_remote_code=True
    ).eval()
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # 多头加载
    ref_param = next(model.parameters())
    dev, dtype = ref_param.device, ref_param.dtype
    hidden = model.config.hidden_size
    num_labels = 3

    if cls_head0_path is None and cls_head_ge1_path is None:
        # 兜底：从 checkpoint 目录尝试附加单头（两层都用同一头）
        print("[cls_head] no external heads provided; falling back to checkpoint search...")
        single_head = nn.Linear(hidden * 2, num_labels, device=dev, dtype=dtype)
        with torch.no_grad():
            if single_head.bias is not None: single_head.bias.zero_()
        model.cls_head = single_head  # for probe api
        # 不强制加载；允许随机初始化（仅用于占位）
        heads = DepthwiseHeads(single_head, single_head)
    else:
        head0 = load_cls_head_from_file(cls_head0_path, hidden, num_labels, dev, dtype) if cls_head0_path else _new_linear_head(hidden, num_labels, dev, dtype)
        head_ge1 = load_cls_head_from_file(cls_head_ge1_path, hidden, num_labels, dev, dtype) if cls_head_ge1_path else head0
        heads = DepthwiseHeads(head0, head_ge1)
        # 为 probe 功能提供一个可用的 cls_head（不参与实际分类）
        model.cls_head = head0

    if probe_cls:
        ok = probe_cls_head_effect(model, tokenizer, device=next(model.parameters()).device)
        if not ok:
            print("[probe] WARNING: probe suggests the attached head may not be active (no noticeable change).")

    df = pd.read_parquet(parquet_path)
    if ("prompt" not in df.columns) and ("messages" not in df.columns):
        raise ValueError("数据集中缺少 'prompt' 或 'messages' 列。")

    tree_depth_gold = defaultdict(lambda: defaultdict(set))
    tree_depth_pred = defaultdict(lambda: defaultdict(set))
    tree_events_gold = defaultdict(list)
    tree_events_pred = defaultdict(list)

    n_rows = min(int(len(df)), int(max_samples))
    print(f"[INFO] evaluating rows: {n_rows}")

    for ridx, row in tqdm(list(df.iloc[:n_rows].iterrows()), total=n_rows, desc="Evaluating"):
        try:
            raw = row.get("prompt", None)
            if raw is None and "messages" in row:
                raw = row["messages"]
            prompt = normalize_prompt(raw)
            if not isinstance(prompt, list) or len(prompt) < 2:
                print(f"[warn] row {ridx}: invalid prompt"); continue

            # system
            sys_prompt = ""
            for m in prompt:
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_prompt = m.get("content", "") or ""
                    break

            # root user
            first_user_idx = None
            for i, m in enumerate(prompt):
                if isinstance(m, dict) and m.get("role") == "user":
                    first_user_idx = i; break
            if first_user_idx is None:
                print(f"[warn] row {ridx}: no user"); continue

            user_str = prompt[first_user_idx].get("content", "")
            try:
                root_user_json = json.loads(user_str) if isinstance(user_str, str) else None
            except Exception:
                root_user_json = None
            if not isinstance(root_user_json, dict):
                print(f"[warn] row {ridx}: user content not json"); continue

            # gold（仅评分）
            rm = root_user_json.get("reward_model") or {}
            gt = rm.get("ground_truth") or {}
            cond_gt_by_turn = gt.get("cond_gt_by_turn") or []
            if not isinstance(cond_gt_by_turn, list):
                cond_gt_by_turn = []

            # record_id
            sft_ci = row.get("sft_chunk_info")
            if isinstance(sft_ci, str):
                try: sft_ci = json.loads(sft_ci)
                except Exception: sft_ci = {}
            record_id = ""
            if isinstance(sft_ci, dict):
                record_id = str(sft_ci.get("record_id") or "")
            if not record_id:
                record_id = f"row{getattr(row, 'name', 'NA')}"

            # BFS init
            q = deque()
            root = dict(root_user_json)
            root.setdefault("historical_interactors", [])
            if not root.get("user_name"):
                root["user_name"] = extract_username_from_content(root.get("content", "") or "") or ROOT_FALLBACK_KEY
            root["_step_depth"] = 0
            root["depth"] = 0
            q.append(root)

            steps = 0
            traj_id = f"{record_id}-traj0"

            while q and steps < max_turns:
                parent_full = q.popleft()
                step_depth = int(parent_full.get("_step_depth", 0))

                preds = classify_and_decode_one_step(
                    model, tokenizer, device=next(model.parameters()).device,
                    system_prompt=sys_prompt, user_json=parent_full,
                    heads=heads,
                    decode=(not no_decode),
                    gen_max_new_tokens=int(gen_max_new_tokens),
                    gen_temperature=float(temperature),
                    gen_top_p=float(gen_top_p),
                    max_input_tokens=int(max_input_tokens),
                    decode_batch_size=int(decode_batch_size),
                    debug=bool(debug_cls),
                )

                # gold（严格父键）
                cond_key = (parent_full.get("user_name") or "").strip() or ROOT_FALLBACK_KEY
                gold_names, ck_out, flags = strict_gold_for_parent(step_depth, cond_key, cond_gt_by_turn)
                gold_set = set(n for n in gold_names if n)

                pred_pos_all = {p["user_name"] for p in preds if int(p.get("type", 0)) in (1, 2) and p.get("user_name")}
                tree_depth_gold[record_id][step_depth].update(gold_set)
                tree_depth_pred[record_id][step_depth].update(pred_pos_all)
                for u in gold_set:
                    tree_events_gold[record_id].append((u, ck_out, step_depth))
                for u in pred_pos_all:
                    tree_events_pred[record_id].append((u, ck_out, step_depth))

                step_f1 = set_f1(list(pred_pos_all), list(gold_set))

                if jsonl_overview_path:
                    overview = {
                        "group_id": record_id,
                        "trajectory_id": traj_id,
                        "depth": step_depth,
                        "input_text": f"SYSTEM:\n{sys_prompt}\n\nUSER:\n{json.dumps(parent_full, ensure_ascii=False)}\n",
                        "output_text": json.dumps(preds, ensure_ascii=False),
                        "parsed_output": sorted(list(pred_pos_all)),
                        "step_reward": float(step_f1),
                    }
                    with open(jsonl_overview_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(overview, ensure_ascii=False) + "\n")

                if jsonl_detail_path:
                    detail = {
                        "ts": int(time.time() * 1000),
                        "group_id": record_id,
                        "trajectory_id": traj_id,
                        "depth": step_depth,
                        "report_depth": step_depth + 1,
                        "cond_key": ck_out,
                        "input_user": parent_full,
                        "output_text": preds,
                        "gold": sorted(list(gold_set)),
                        "gold_level_mode": flags.get("mode"),
                        "gold_used_list_level": bool(flags.get("used_list_level")),
                        "gold_parent_missing": bool(flags.get("parent_missing")),
                        "gold_depth_out_of_range": bool(flags.get("depth_out_of_range")),
                        "gold_empty_after_parse": bool(flags.get("empty_after_parse")),
                        "used_fallback": bool(flags.get("fallback_to_empty")),
                        "parse_status": "ok",
                        "step_reward": float(step_f1),
                        "finish_reason": "stop",
                    }
                    with open(jsonl_detail_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(detail, ensure_ascii=False) + "\n")

                # === BFS 下一层：重渲染 POTENTIAL_SPANS（depth+1），并携带 root_potential 给下一步重建 ===
                if step_depth < depth_limit:
                    parent_name = (parent_full.get("user_name") or "").strip()
                    pots = get_root_potential_objs(parent_full) or get_root_potential_objs(root_user_json)
                    base_content = strip_psep_block(parent_full.get("content", "") or "")
                    next_depth = step_depth + 1
                    if pots:
                        next_content = (base_content.rstrip() + render_psep_block_from_list(pots, depth=next_depth))
                    else:
                        next_content = bump_potential_depths(parent_full.get("content", ""), inc=1)

                    for it in preds:
                        if int(it.get("type", 0)) == 1:  # 仅评论入队
                            uname = (it.get("user_name") or "").strip()
                            child = {
                                "user_name": uname,
                                "content": next_content,
                                "historical_interactors": [{"user_name": parent_name}],
                                "_step_depth": next_depth,
                                "depth": next_depth,
                                "reward_model": {
                                    "ground_truth": {"cond_gt_by_turn": cond_gt_by_turn},
                                    "root_potential": (root_user_json.get("reward_model", {}) or {}).get("root_potential")
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

    # 分层并集 F1
    all_depths = sorted({
        d for rid, depth2gold in tree_depth_gold.items() for d in depth2gold.keys()
    } | {
        d for rid, depth2pred in tree_depth_pred.items() for d in depth2pred.keys()
    })
    per_depth_f1_values = defaultdict(list)
    all_f1_across_tree_layers = []
    for rid in tree_depth_gold.keys() | tree_depth_pred.keys():
        depth2gold = tree_depth_gold.get(rid, {})
        depth2pred = tree_depth_pred.get(rid, {})
        for d in all_depths:
            gset = set(depth2gold.get(d, set()))
            pset = set(depth2pred.get(d, set()))
            f1 = set_f1(list(pset), list(gset))
            per_depth_f1_values[d].append(f1)
            all_f1_across_tree_layers.append(f1)

    print("\n=== 分层平均 F1（树内合并后，再跨树平均） ===")
    if not per_depth_f1_values:
        print("(无可统计项)")
    else:
        for d in sorted(per_depth_f1_values.keys()):
            depth_label = d + 1
            vals = per_depth_f1_values[d]
            mean_f1 = float(np.mean(vals)) if len(vals) > 0 else 0.0
            print(f"depth={depth_label}: mean_F1={mean_f1:.4f}  (trees={len(vals)})")
    overall_f1 = float(np.mean(all_f1_across_tree_layers)) if len(all_f1_across_tree_layers) > 0 else 0.0
    print(f"\n=== 总 F1 ===\noverall_mean_F1={overall_f1:.4f}  (count={len(all_f1_across_tree_layers)})")

    # 图评估
    per_tree_rel = []
    all_rids = set(tree_events_gold.keys()) | set(tree_events_pred.keys())
    for rid in sorted(all_rids):
        gold_ev = tree_events_gold.get(rid, [])
        pred_ev = tree_events_pred.get(rid, [])
        cnt_gold = edge_counts_from_events(gold_ev, directed=graph_directed)
        cnt_pred = edge_counts_from_events(pred_ev, directed=graph_directed)
        edges_all = sorted(set(cnt_gold.keys()) | set(cnt_pred.keys()))
        if len(edges_all) == 0: continue
        a = np.array([cnt_gold.get(e, 0.0) for e in edges_all], dtype=float)
        b = np.array([cnt_pred.get(e, 0.0) for e in edges_all], dtype=float)
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))
        per_tree_rel.append(rel)

    if per_tree_rel:
        mean_rel = float(np.mean(per_tree_rel))
        print("\n=== 图评估（树级合并后） ===")
        print(f"Trees={len(per_tree_rel)}  mean_rel_error={mean_rel:.4f}")
    else:
        print("\n=== 图评估（树级合并后） ===")
        print("(无可统计项)")

# ===== CLI =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO parquet with BFS rollout using depth-wise classification heads + full-context decoding (no gold leakage)."
    )
    parser.add_argument("--data", type=str, required=True, help="输入 GRPO Parquet 文件（含 prompt）")
    parser.add_argument("--model", type=str, required=True, help="基础模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer 路径（未提供则回退到 --model）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")
    parser.add_argument("--max_samples", type=int, default=600, help="最大样本条数")
    parser.add_argument("--max_turns", type=int, default=20, help="每个对话的最大展开步数")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度（仅对评论生效）")
    parser.add_argument("--depth_limit", type=int, default=1, help="展开深度上限（0基）")
    parser.add_argument("--jsonl_overview", type=str, default="rollout_overview.jsonl", help="概览 JSONL 输出路径（追加写）")
    parser.add_argument("--jsonl_detail", type=str, default="rollout_io_gold.jsonl", help="明细 JSONL 输出路径（追加写）")
    parser.add_argument("--report_path", type=str, default=None, help="最终汇总结果 TXT 输出路径（追加写）")
    parser.add_argument("--undirected_graph", action="store_true", help="若设定，则图评估按无向边计权（默认有向）")

    # 生成相关
    parser.add_argument("--gen_max_new_tokens", type=int, default=48, help="评论生成的最大 token 数")
    parser.add_argument("--gen_top_p", type=float, default=0.9, help="nucleus sampling 的 p")
    parser.add_argument("--no_decode", action="store_true", help="仅分类不生成评论（用于纯分类评估）")
    parser.add_argument("--decode_batch_size", type=int, default=8, help="生成阶段的小批量大小")

    # 调试/探针
    parser.add_argument("--debug_cls", action="store_true", help="打印每候选 softmax 概率与所用分类头")
    parser.add_argument("--assert_cls_loaded", action="store_true", help="（保留参数位，不强制）")
    parser.add_argument("--probe_cls_head", action="store_true", help="在线探针：打乱 cls_head 验证其是否起作用（仅对 head0）")

    # 截断控制
    parser.add_argument("--max_input_tokens", type=int, default=8192, help="编码/生成输入的最大长度（左截断）")

    # 多头路径（新增）
    parser.add_argument("--cls_head0", dest="cls_head0", type=str, default=None, help="depth==0 使用的分类头 .pt")
    parser.add_argument("--cls_head_ge1", dest="cls_head_ge1", type=str, default=None, help="depth>=1 使用的分类头 .pt")

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
        jsonl_overview_path=args.jsonl_overview,
        jsonl_detail_path=args.jsonl_detail,
        report_path=args.report_path,
        graph_directed=(not args.undirected_graph),
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_top_p=args.gen_top_p,
        no_decode=args.no_decode,
        debug_cls=args.debug_cls,
        assert_cls_loaded=args.assert_cls_loaded,
        probe_cls=args.probe_cls_head,
        max_input_tokens=args.max_input_tokens,
        decode_batch_size=args.decode_batch_size,
        cls_head0_path=args.cls_head0,
        cls_head_ge1_path=args.cls_head_ge1,
    )
