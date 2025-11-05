#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从训练日志解析并绘制两个分类头 loss 变化曲线。

功能：
1) 逐行解析日志，抽取 epoch、step、train/cls0_loss、train/cls1_loss、train/cls_loss
2) 支持移动平均平滑（--smooth）
3) 输出 PNG 曲线图与对齐好的明细 CSV

兼容的日志行示例：
Epoch 1/3:   0%|          | 1/417 [00:20<2:21:02, 20.34s/it]step:2 - train/loss:0.009107 - train/lr(1e-3):0.005000 - train/cls_weight:0.800000 - train/gen_weight:0.200000 - train/cls_loss:0.048563 - train/cls0_loss:0.000000 - train/cls1_loss:0.048563 - ...

使用示例：
python plot_cls_losses.py -i train.log -o cls_losses.png --csv cls_losses.csv --smooth 5
"""

import re
import os
import argparse
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

# ---- 正则准备（尽量鲁棒，兼容科学计数法）----
FLOAT = r'[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?'
PAT_EPOCH = re.compile(r'Epoch\s+(\d+)\s*/\s*(\d+)')
PAT_STEP  = re.compile(r'\bstep:(\d+)\b')
# 捕获形如 "train/cls0_loss:0.123456"
PAT_METRIC = re.compile(r'\b(train|eval)/(cls0_loss|cls1_loss|cls_loss):(' + FLOAT + r')\b')

def read_text_lines(path: str) -> List[str]:
    # 兼容 Windows 控制台日志编码
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                return f.readlines()
        except Exception:
            continue
    # 最后兜底
    with open(path, 'r', errors='ignore') as f:
        return f.readlines()

def parse_log(path: str, pick_split: str = "train") -> pd.DataFrame:
    """
    解析单个日志文件，返回包含：
    ['file','seq','epoch','epoch_total','step_in_epoch','cls0_loss','cls1_loss','cls_loss']
    的 DataFrame；seq 为按出现顺序的全局序号（从 1 开始）。
    仅保留同时出现任意一个目标 metric 的样本行（优先 train/*）。
    """
    lines = read_text_lines(path)
    seq = 0
    cur_epoch = None
    cur_epoch_total = None

    rows: List[Dict[str, Any]] = []

    for ln in lines:
        # 更新 epoch
        m_epoch = PAT_EPOCH.search(ln)
        if m_epoch:
            cur_epoch = int(m_epoch.group(1))
            cur_epoch_total = int(m_epoch.group(2))

        # 解析 step
        m_step = PAT_STEP.search(ln)
        step_in_epoch = int(m_step.group(1)) if m_step else None

        # 抓取这一行内所有 metric
        metrics = PAT_METRIC.findall(ln)  # list of tuples: [(split, name, value_str), ...]
        if not metrics:
            continue

        # 优先取 train/*（如果 pick_split=="train"），否则 eval/*
        # 但也允许同一行同时有多个，分别挑出需要的
        bucket: Dict[str, float] = {}
        for split, name, val in metrics:
            if split != pick_split:
                continue
            try:
                bucket[name] = float(val)
            except ValueError:
                continue

        # 如果优先 split 没抓到，再尝试另一 split 兜底（可选）
        if not bucket:
            for split, name, val in metrics:
                # 兜底抓任何 split（通常是 eval）
                try:
                    bucket.setdefault(name, float(val))
                except ValueError:
                    pass

        # 只在至少抓到一个目标 loss 时记录
        keys_of_interest = {"cls0_loss", "cls1_loss", "cls_loss"}
        if keys_of_interest.intersection(bucket.keys()):
            seq += 1
            rows.append({
                "file": os.path.basename(path),
                "seq": seq,
                "epoch": cur_epoch,
                "epoch_total": cur_epoch_total,
                "step_in_epoch": step_in_epoch,
                "cls0_loss": bucket.get("cls0_loss"),
                "cls1_loss": bucket.get("cls1_loss"),
                "cls_loss": bucket.get("cls_loss"),
            })

    df = pd.DataFrame(rows)
    return df

def concat_with_source(files: List[str], pick_split: str = "train") -> pd.DataFrame:
    dfs = []
    for fp in files:
        df = parse_log(fp, pick_split=pick_split)
        if df.empty:
            print(f"[warn] {fp} 未解析到目标指标。")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # 增加一个“全局步”列：按文件内 seq 连续，跨文件不连续；为了可视化清晰，这里再做一个跨文件连续序号
    out["global_seq"] = range(1, len(out) + 1)
    return out

def apply_smoothing(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=False).mean()

def main():
    ap = argparse.ArgumentParser(description="从训练日志绘制两个分类头 loss 曲线")
    ap.add_argument("-i", "--inputs", nargs="+", required=True, help="一个或多个日志文件路径")
    ap.add_argument("-o", "--out", default="cls_losses.png", help="输出曲线图 PNG 路径")
    ap.add_argument("--csv", default="cls_losses.csv", help="导出的 CSV 路径")
    ap.add_argument("--smooth", type=int, default=1, help="平滑窗口大小(移动平均)，默认为1=不平滑")
    ap.add_argument("--split", choices=["train", "eval"], default="train", help="优先解析的 split（默认 train）")
    ap.add_argument("--figsize", type=float, nargs=2, default=(10, 5), help="图尺寸，默认 10 5")
    args = ap.parse_args()

    df = concat_with_source(args.inputs, pick_split=args.split)
    if df.empty:
        print("[error] 没有可用数据，检查日志路径与内容。")
        return

    # 对每个文件分别做平滑，避免跨文件相互影响
    df = df.sort_values(["file", "seq"]).reset_index(drop=True)
    df["cls0_loss_s"] = df.groupby("file")["cls0_loss"].transform(lambda x: apply_smoothing(x, args.smooth))
    df["cls1_loss_s"] = df.groupby("file")["cls1_loss"].transform(lambda x: apply_smoothing(x, args.smooth))
    df["cls_loss_s"]  = df.groupby("file")["cls_loss" ].transform(lambda x: apply_smoothing(x, args.smooth))

    # 保存 CSV
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"[ok] 已导出 CSV -> {args.csv}")

    # 绘图（按文件区分颜色/线型）
    plt.figure(figsize=tuple(args.figsize), dpi=150)
    # 为了图例清晰：每个文件画两条（cls0 & cls1）；cls_loss 画成细线参考
    for fname, sub in df.groupby("file", sort=False):
        x = sub["global_seq"]
        # 只在有值时绘制
        if sub["cls0_loss_s"].notna().any():
            plt.plot(x, sub["cls0_loss_s"], label=f"{fname} • cls0_loss")
        if sub["cls1_loss_s"].notna().any():
            plt.plot(x, sub["cls1_loss_s"], label=f"{fname} • cls1_loss")
        # 参考总分类 loss（如果存在）
        if sub["cls_loss_s"].notna().any():
            plt.plot(x, sub["cls_loss_s"], linestyle="--", linewidth=1, label=f"{fname} • cls_loss(ref)")

    title = f"Classification Head Losses ({args.split})"
    if args.smooth and args.smooth > 1:
        title += f"  |  moving avg={args.smooth}"
    plt.title(title)
    plt.xlabel("Global sequence (by appearance)")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"[ok] 已保存曲线图 -> {args.out}")

if __name__ == "__main__":
    main()
