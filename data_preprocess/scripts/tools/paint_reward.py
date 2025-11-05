#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import csv
import argparse
import math
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
    # 去掉 ANSI 转义序列（含彩色 PID 前缀等）
    return ANSI_RE.sub('', s)

def parse_progress_segment_means(log_path: str, keep_incomplete: bool = False):
    """
    根据 'Training Progress: ...' 行划分区间：
    - 每两条进度行之间，收集 gold 非空的 [rollout][kept-step] 的 step_F1(raw) 作为 step_reward
    - 对该区间内的 step_reward 求均值，得到一个点
    - 若最后一段没有闭合：默认丢弃；--keep-incomplete 时保留

    返回：按区间顺序的点列表，每点包含：
    {
      "point_idx": int,                  # 1-based
      "progress_text": str,              # 结束该段的进度行文本（无 ANSI）
      "timestamp": str,                  # 结束该段进度行里的时间戳（若解析到）
      "n_rewards": int,                  # 本段用于均值的样本数
      "mean_reward": float or nan,       # 均值
    }
    """
    # 进度条行：宽松匹配
    progress_pat = re.compile(r"Training Progress:\s+.+")
    # 时间戳：如 [2025-09-17 13:01:13] 或 [2025-09-17 13:01:13,233]
    ts_pat = re.compile(r"\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\]")
    # kept-step 行：抓 step_F1(raw) 与 gold 列表
    kept_step_pat = re.compile(
        r"\[rollout\]\[kept-step\].*?"
        r"step_F1\(raw\)=([-\d\.eE]+).*?"
        r"gold=\[([^\]]*)\]",
        re.DOTALL
    )

    points = []
    current_rewards = []
    have_open_segment = False  # 是否已经见到第一个进度行（开始收集）
    last_progress_text = ""
    last_progress_ts = ""

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw)

            # 命中进度行：关闭上一段 -> 产出一个点；然后开启新段
            if progress_pat.search(line):
                # 先收尾上一段（仅当之前已开始收集）
                if have_open_segment:
                    mean_val = float("nan")
                    if len(current_rewards) > 0:
                        mean_val = sum(current_rewards) / len(current_rewards)
                    points.append({
                        "point_idx": len(points) + 1,
                        "progress_text": last_progress_text,
                        "timestamp": last_progress_ts,
                        "n_rewards": len(current_rewards),
                        "mean_reward": mean_val,
                    })
                    current_rewards = []

                # 开启新段，记录“上一段的结束进度行文本/时间戳”
                have_open_segment = True
                last_progress_text = line.strip()
                ts_m = ts_pat.search(line)
                last_progress_ts = ts_m.group(1) if ts_m else ""

                # 进入下一轮收集（直到遇到下一条进度行）
                continue

            # 在进度段内，匹配 kept-step 行并提取 reward（gold 非空才加入）
            if have_open_segment:
                m = kept_step_pat.search(line)
                if m:
                    reward = float(m.group(1))
                    gold_inside = m.group(2)  # gold 方括号内部的内容
                    # gold 非空判断：[] -> 空；否则认为非空
                    if gold_inside.strip() != "":
                        current_rewards.append(reward)

    # 文件结束：如果最后一段未闭合
    if have_open_segment and (keep_incomplete or len(points) == 0):
        # 用最后一次见到的 progress 作为“该段的结束”
        mean_val = float("nan")
        if len(current_rewards) > 0:
            mean_val = sum(current_rewards) / len(current_rewards)
        points.append({
            "point_idx": len(points) + 1,
            "progress_text": last_progress_text,
            "timestamp": last_progress_ts,
            "n_rewards": len(current_rewards),
            "mean_reward": mean_val,
        })

    return points

def write_csv(csv_path: str, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_idx", "timestamp", "n_rewards", "mean_reward", "progress_text"])
        for r in rows:
            mean_str = f'{r["mean_reward"]:.6f}' if not math.isnan(r["mean_reward"]) else "nan"
            w.writerow([r["point_idx"], r["timestamp"], r["n_rewards"], mean_str, r["progress_text"]])

def plot_points(rows, out_path: str):
    if not rows:
        print("[WARN] 没有可绘制的数据点。")
        return
    xs = [r["point_idx"] for r in rows]
    ys = [r["mean_reward"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o", label="Avg step_reward between progress lines")
    plt.title("Mean step_reward per segment (gold != [])")
    plt.xlabel("Segment index (between successive 'Training Progress' lines)")
    plt.ylabel("Mean step_reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[+] Saved plot to {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log",  required=True, help="路径到日志/jsonl（包含 Training Progress 与 [rollout][kept-step] 行）")
    p.add_argument("--csv",  required=True, help="输出 CSV 路径")
    p.add_argument("--plot", required=True, help="输出 PNG 路径")
    p.add_argument("--keep-incomplete", action="store_true",
                   help="保留最后一个未被下一条进度行闭合的尾段（默认丢弃）")
    args = p.parse_args()

    rows = parse_progress_segment_means(args.log, keep_incomplete=args.keep_incomplete)
    if not rows:
        print(f"[WARN] 在 {args.log} 中未形成任何分段均值（检查是否存在两条以上 'Training Progress' 行及符合条件的 kept-step 行）。")
        return

    write_csv(args.csv, rows)
    print(f"[+] Wrote CSV to {args.csv}")
    plot_points(rows, args.plot)

if __name__ == "__main__":
    main()
