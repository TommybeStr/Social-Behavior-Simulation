#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from statistics import mean

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub('', s)

# 匹配 [rollout][kept-step] 行
LINE_RE = re.compile(
    r"\[rollout\]\[kept-step\].*?"
    r"depth=(\d+)\s+"
    r"status=([a-z_]+)\s+"
    r"step_F1\(raw\)=([-\d\.eE]+)\s+"
    r"cond_keys=(\[[^\]]*\])\s+"
    r"gold=(\[[^\]]*\])\s+"
    r"pred=(\[[^\]]*\])",
    re.IGNORECASE
)

def parse_json_array(s: str):
    """把形如 ["a","b"] 的字符串解析成 list[str]，失败则用兜底提取双引号内容。"""
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return re.findall(r'"([^"]*)"', s)

def analyze(log_path: str):
    # 原有统计
    depth0_f1_all = []
    depthN_f1_all = []
    gold_empty_total = 0
    gold_empty_pred_nonempty = 0
    pred_empty_total = 0
    pred_empty_gold_nonempty = 0

    # 新增：排除 parse_fail 后的统计
    depth0_f1_ok = []
    depthN_f1_ok = []

    # depth!=0 & gold!=[] 的统计
    depthN_gold_nonempty_f1_all = []
    depthN_gold_nonempty_f1_ok = []   # ⬅️ 新增（排除 parse_fail）

    matched = 0

    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = strip_ansi(raw)
            m = LINE_RE.search(line)
            if not m:
                continue
            matched += 1
            depth = int(m.group(1))
            status = (m.group(2) or "").lower()
            f1 = float(m.group(3))
            gold_str = m.group(5)
            pred_str = m.group(6)

            gold = parse_json_array(gold_str)
            pred = parse_json_array(pred_str)

            # depth 分组
            if depth == 0:
                depth0_f1_all.append(f1)
            else:
                depthN_f1_all.append(f1)

            # 去除 parse_fail
            if status != "parse_fail":
                if depth == 0:
                    depth0_f1_ok.append(f1)
                else:
                    depthN_f1_ok.append(f1)

            # gold==[] 相关统计
            if len(gold) == 0:
                gold_empty_total += 1
                if len(pred) > 0:
                    gold_empty_pred_nonempty += 1

            # pred==[] 相关统计
            if len(pred) == 0:
                pred_empty_total += 1
                if len(gold) > 0:
                    pred_empty_gold_nonempty += 1

            # depth!=0 & gold!=[] （全量 + 去除 parse_fail）
            if depth != 0 and len(gold) > 0:
                depthN_gold_nonempty_f1_all.append(f1)
                if status != "parse_fail":
                    depthN_gold_nonempty_f1_ok.append(f1)

    def safe_mean(x):
        return float(mean(x)) if x else float('nan')

    # 均值
    depth0_mean_all = safe_mean(depth0_f1_all)
    depthN_mean_all = safe_mean(depthN_f1_all)
    depth0_mean_ok = safe_mean(depth0_f1_ok)
    depthN_mean_ok = safe_mean(depthN_f1_ok)

    depthN_gold_nonempty_mean_all = safe_mean(depthN_gold_nonempty_f1_all)
    depthN_gold_nonempty_mean_ok = safe_mean(depthN_gold_nonempty_f1_ok)

    # 占比
    ratio_gold_empty_pred_nonempty = (
        gold_empty_pred_nonempty / gold_empty_total if gold_empty_total > 0 else float('nan')
    )
    ratio_pred_empty_gold_nonempty = (
        pred_empty_gold_nonempty / pred_empty_total if pred_empty_total > 0 else float('nan')
    )

    # 输出
    print("=== Kept-Step Metrics ===")
    print(f"matched_lines                                 : {matched}")
    print(f"depth=0 count (all)                           : {len(depth0_f1_all)}")
    print(f"depth!=0 count (all)                          : {len(depthN_f1_all)}")
    print(f"1) depth=0 mean step_F1(raw) (all)            : {depth0_mean_all:.6f}" if depth0_f1_all else "1) depth=0 mean step_F1(raw) (all)            : n/a")
    print(f"2) depth!=0 mean step_F1(raw) (all)           : {depthN_mean_all:.6f}" if depthN_f1_all else "2) depth!=0 mean step_F1(raw) (all)           : n/a")

    print(f"1a) depth=0 count (status!=parse_fail)        : {len(depth0_f1_ok)}")
    print(f"1b) depth=0 mean step_F1(raw) (status!=parse_fail): {depth0_mean_ok:.6f}" if depth0_f1_ok else "1b) depth=0 mean step_F1(raw) (status!=parse_fail): n/a")
    print(f"2a) depth!=0 count (status!=parse_fail)       : {len(depthN_f1_ok)}")
    print(f"2b) depth!=0 mean step_F1(raw) (status!=parse_fail): {depthN_mean_ok:.6f}" if depthN_f1_ok else "2b) depth!=0 mean step_F1(raw) (status!=parse_fail): n/a")

    print(f"3) among gold==[]: pred!=[] ratio             : "
          f"{ratio_gold_empty_pred_nonempty:.4f} ({gold_empty_pred_nonempty}/{gold_empty_total})"
          if gold_empty_total > 0 else
          "3) among gold==[]: pred!=[] ratio             : n/a (no gold==[] steps)")
    print(f"4) among pred==[]: gold!=[] ratio             : "
          f"{ratio_pred_empty_gold_nonempty:.4f} ({pred_empty_gold_nonempty}/{pred_empty_total})"
          if pred_empty_total > 0 else
          "4) among pred==[]: gold!=[] ratio             : n/a (no pred==[] steps)")

    # 新增输出项
    print(f"5) depth!=0 & gold!=[] count (all)            : {len(depthN_gold_nonempty_f1_all)}")
    print(f"6) depth!=0 & gold!=[] mean step_F1(raw) (all): "
          f"{depthN_gold_nonempty_mean_all:.6f}" if depthN_gold_nonempty_f1_all else
          "6) depth!=0 & gold!=[] mean step_F1(raw) (all): n/a")
    print(f"7) depth!=0 & gold!=[] count (status!=parse_fail)       : {len(depthN_gold_nonempty_f1_ok)}")
    print(f"8) depth!=0 & gold!=[] mean step_F1(raw) (status!=parse_fail): "
          f"{depthN_gold_nonempty_mean_ok:.6f}" if depthN_gold_nonempty_f1_ok else
          "8) depth!=0 & gold!=[] mean step_F1(raw) (status!=parse_fail): n/a")

def main():
    ap = argparse.ArgumentParser(description="Parse [rollout][kept-step] lines and compute metrics.")
    ap.add_argument("--log", required=True, help="路径到控制台日志文件")
    args = ap.parse_args()
    analyze(args.log)

if __name__ == "__main__":
    main()
