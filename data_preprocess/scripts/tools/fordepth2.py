# -*- coding: utf-8 -*-
import json
import argparse
import re
from typing import Iterator, Dict, Any

# 与评估脚本一致的“无交互”文案
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"

# 宽松/严格头部都可，这里用严格口径即可
_INTERACT_HEAD_RE = re.compile(
    r'^\s*\[\s*INTERACT\s*=\s*([01])\s*\]\s*(?:\r?\n)?',
    re.IGNORECASE
)

def _peel_interact_header(s: str):
    """剥离 [INTERACT=0/1] 头；返回 (tag, body)。"""
    if not isinstance(s, str):
        return None, ""
    m = _INTERACT_HEAD_RE.match(s)
    if not m:
        return None, s.strip()
    tag = int(m.group(1))
    body = s[m.end():].strip()
    return tag, body

def _has_pred(output_text: str) -> bool:
    """
    判定“有 pred”：
    - 若头为 [INTERACT=0] → False
    - 否则主体需为非空 JSON 数组，且不等于固定“无交互”文案
    """
    try:
        s = (output_text or "").strip()
        if not s:
            return False
        tag, body = _peel_interact_header(s)
        if tag == 0:
            return False
        if tag == 1:
            s = body
        if not s or s == "[]" or s == NO_INTERACTION_STR:
            return False
        obj = json.loads(s)
        return isinstance(obj, list) and len(obj) > 0
    except Exception:
        return False

def iter_detail_records(path: str) -> Iterator[Dict[str, Any]]:
    """读取 detail 日志：支持 JSONL 或 JSON 数组。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            # JSON 数组
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            yield rec
                return
            except Exception:
                pass
        # JSONL
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec

def main():
    ap = argparse.ArgumentParser(
        description=(
            "从评估 detail 日志中抽取：report_depth==2 且“有 pred”的记录；"
            "输出仅包含 { user: <input_user>, output_text: <原样输出> }。"
        )
    )
    ap.add_argument("--input", required=True, help="detail JSONL/JSON 数组 文件路径（如 rollout_io_gold.jsonl）")
    ap.add_argument("--output", required=True, help="输出文件路径")
    ap.add_argument("--out_format", choices=["jsonl", "json"], default="jsonl",
                    help="输出格式：jsonl（默认）或 json（数组）")
    args = ap.parse_args()

    results = []
    total = 0
    kept = 0
    bad_depth = 0
    missing_user = 0

    for rec in iter_detail_records(args.input):
        total += 1

        # 只看第二层（业务口径，detail 里已写为 step_depth+1）
        rd = rec.get("report_depth", None)
        try:
            rd = int(rd)
        except Exception:
            bad_depth += 1
            continue
        if rd != 2:
            continue

        out_text = rec.get("output_text", "")
        if not _has_pred(out_text):
            continue

        user_obj = rec.get("input_user", None)  # detail 日志里就有清洗后的 user
        if not isinstance(user_obj, dict):
            missing_user += 1
            continue

        results.append({
            "user": user_obj,
            "output_text": out_text
        })
        kept += 1

    # 写出
    if args.out_format == "jsonl":
        with open(args.output, "w", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"[done] total_seen={total}, kept={kept}, skipped_bad_depth={bad_depth}, missing_input_user={missing_user}")
    print(f"[write] saved to: {args.output} (format={args.out_format})")

if __name__ == "__main__":
    main()
