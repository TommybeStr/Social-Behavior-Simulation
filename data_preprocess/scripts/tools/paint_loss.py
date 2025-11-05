#!/usr/bin/env python3
import re
import csv
import argparse
import matplotlib.pyplot as plt

def strip_ansi(s):
    # 去掉 ANSI 转义序列
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', s)

def parse_log(log_path):
    steps = []
    losses = []
    scores = []
    # 匹配 step、pg_loss、critic/score/mean
    pattern = re.compile(
        r"step:(\d+).*?actor/pg_loss:([-\d\.eE]+).*?critic/score/mean:([-\d\.eE]+)"
    )
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = strip_ansi(raw)
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                score = float(m.group(3))
                steps.append(step)
                losses.append(loss)
                scores.append(score)
    return steps, losses, scores

def write_csv(csv_path, steps, losses, scores):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'f1_score'])
        for s, l, sc in zip(steps, losses, scores):
            writer.writerow([s, l, sc])

def plot(steps, losses, scores, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker='o', label='Loss')
    plt.plot(steps, scores, marker='x', label='F1 Score')
    plt.title('Loss & F1 over Steps')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[+] Saved plot to {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log',    required=True, help='路径到 .jsonl / console 日志')
    p.add_argument('--csv',    required=True, help='输出 CSV 路径')
    p.add_argument('--plot',   required=True, help='输出 PNG 路径')
    args = p.parse_args()

    steps, losses, scores = parse_log(args.log)
    if not steps:
        print(f"[WARN] 在 {args.log} 中没匹配到任何 step/pg_loss/critic/score/mean，"
              "请确认日志里是否含有类似 “step:1 … actor/pg_loss:-0.910 … critic/score/mean:0.881” 的行")
        return

    write_csv(args.csv, steps, losses, scores)
    print(f"[+] Wrote metrics CSV to {args.csv}")
    plot(steps, losses, scores, args.plot)

if __name__ == '__main__':
    main()
