#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从训练日志解析并绘制 micro-batch 级别的 head loss / content loss / total loss 曲线。

支持的日志格式：
1) [micro] 1/4 | cls0=0.2199 cls1=0.0000 content=0.0000 | depth0=1 depth1=0 pairs=16
2) step:136 - train/loss:0.197889 - train/cls0_loss:0.219877 - ...（包含 train/loss）

关键逻辑更新：
- Head0 绘图：只要 depth0 == 1，就绘制（无论 cls0_loss 是否为 0）
- Head1 绘图：只要 depth1 == 1，就绘制
- Total loss：直接使用日志中的 train/loss 字段

输出 4 张 PNG：
  <prefix>_micro_head0.png
  <prefix>_micro_head1.png
  <prefix>_micro_content.png
  <prefix>_micro_total.png

用法示例：
python plot_micro_losses.py -i train.log -o run1_micro --smooth 50
"""

import re
import os
import argparse
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

# ---- 正则（兼容科学计数法）----
FLOAT = r'[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?'

# Epoch 行
PAT_EPOCH = re.compile(r'Epoch\s+(\d+)\s*/\s*(\d+)')

# Train loss 行（包含 step 和 train/loss）
PAT_TRAIN_LOSS = re.compile(
    r'step:(\d+)\s*-\s*'
    r'train/loss:(' + FLOAT + r')\s*-\s*'
    r'train/cls_weight:(' + FLOAT + r')\s*-\s*'
    r'train/gen_weight:(' + FLOAT + r')\s*-\s*'
    r'train/cls_loss:(' + FLOAT + r')\s*-\s*'
    r'train/cls0_loss:(' + FLOAT + r')\s*-\s*'
    r'train/cls1_loss:(' + FLOAT + r')\s*-\s*'
    r'train/content_loss:(' + FLOAT + r')'
)

# Micro 行：
# [micro] 1/4 | cls0=0.2199 cls1=0.0000 content=0.0000 | depth0=1 depth1=0 pairs=16
PAT_MICRO = re.compile(
    r'\[micro\]\s+(\d+)/(\d+)\s*\|\s*'
    r'cls0=(' + FLOAT + r')\s+'
    r'cls1=(' + FLOAT + r')\s+'
    r'content=(' + FLOAT + r')\s*\|\s*'
    r'depth0=(\d+)\s+depth1=(\d+)\s+pairs=(\d+)'
)


def read_text_lines(path: str) -> List[str]:
    """尽量容错地读取文本行。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                return f.readlines()
        except Exception:
            continue
    with open(path, 'r', errors='ignore') as f:
        return f.readlines()


def parse_micro_log(path: str) -> pd.DataFrame:
    lines = read_text_lines(path)

    cur_epoch = None
    cur_epoch_total = None
    cur_step_in_epoch = None
    train_loss_by_step = {}  # step_num -> total_loss

    rows: List[Dict[str, Any]] = []
    micro_global_seq = 0

    for ln in lines:
        # 更新 epoch
        m_epoch = PAT_EPOCH.search(ln)
        if m_epoch:
            cur_epoch = int(m_epoch.group(1))
            cur_epoch_total = int(m_epoch.group(2))

        # 解析 train/loss 行（包含 step）
        m_train = PAT_TRAIN_LOSS.search(ln)
        if m_train:
            step_num = int(m_train.group(1))
            total_loss_val = float(m_train.group(2))
            train_loss_by_step[step_num] = total_loss_val
            cur_step_in_epoch = step_num

        # 解析 micro 行
        m = PAT_MICRO.search(ln)
        if not m:
            continue

        micro_global_seq += 1
        micro_idx      = int(m.group(1))
        micro_total    = int(m.group(2))
        cls0_val       = float(m.group(3))
        cls1_val       = float(m.group(4))
        content_val    = float(m.group(5))
        depth0_cnt     = int(m.group(6))
        depth1_cnt     = int(m.group(7))
        pairs_cnt      = int(m.group(8))

        total_loss_val = train_loss_by_step.get(cur_step_in_epoch, None)

        row = {
            "file": os.path.basename(path),
            "micro_global_seq": micro_global_seq,
            "micro_idx_in_step": micro_idx,
            "micro_total_in_step": micro_total,
            "epoch": cur_epoch,
            "epoch_total": cur_epoch_total,
            "step_in_epoch": cur_step_in_epoch,
            "cls0_loss": cls0_val,
            "cls1_loss": cls1_val,
            "content_loss": content_val,
            "total_loss": total_loss_val,
            "depth0": depth0_cnt,
            "depth1": depth1_cnt,
            "pairs": pairs_cnt,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def concat_with_source(files: List[str]) -> pd.DataFrame:
    dfs = []
    for fp in files:
        df = parse_micro_log(fp)
        if df.empty:
            print(f"[warn] {fp} 未解析到任何 [micro] 行。")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["row_seq"] = range(1, len(out) + 1)
    return out


def apply_smoothing(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def plot_head_micro(df: pd.DataFrame, head_id: int, out_png: str,
                    smooth: int, xcol: str, title_extra: str):
    """
    对 head0/head1：根据 depth0/depth1 判断是否属于该 head。
    - head0: depth0 == 1
    - head1: depth1 == 1
    即使 loss=0 也绘制。
    """
    plt.figure(figsize=(10, 5), dpi=150)
    key_loss = f"cls{head_id}_loss"
    depth_col = "depth0" if head_id == 0 else "depth1"

    missing_cols = []
    if key_loss not in df.columns:
        missing_cols.append(key_loss)
    if depth_col not in df.columns:
        missing_cols.append(depth_col)
    if missing_cols:
        print(f"[warn] 缺少列 {missing_cols}，跳过绘制 {out_png}")
        plt.close()
        return

    df_hit = df[df[depth_col] == 1].copy()
    if df_hit.empty:
        plt.title(f"Head-{head_id} micro loss (no micro with {depth_col}=1)")
        plt.xlabel("Micro index")
        plt.ylabel(key_loss)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[ok] Head-{head_id} 图已保存 -> {out_png} (0 点)")
        return

    if xcol not in df_hit.columns:
        xcol = "row_seq"

    df_hit = df_hit.sort_values(xcol)
    df_hit["plot_x"] = range(1, len(df_hit) + 1)
    df_hit[key_loss] = apply_smoothing(df_hit[key_loss], smooth)

    total_points = len(df_hit)

    for fname, sub in df_hit.groupby("file", sort=False):
        plt.plot(sub["plot_x"], sub[key_loss], label=f"{fname} (n={len(sub)})")

    ttl = f"Head-{head_id} micro loss (filtered by {depth_col}=1)"
    if title_extra:
        ttl += f" | {title_extra}"
    if smooth > 1:
        ttl += f" | MA={smooth}"
    plt.title(ttl)
    plt.xlabel("Micro index (1..N)")
    plt.ylabel(key_loss)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[ok] Head-{head_id} 图已保存 -> {out_png} (共 {total_points} 点)")


def plot_content_micro(df: pd.DataFrame, out_png: str,
                       smooth: int, xcol: str, title_extra: str):
    plt.figure(figsize=(10, 5), dpi=150)
    key_loss = "content_loss"

    if key_loss not in df.columns:
        print(f"[warn] 缺少列 {key_loss}，跳过绘制 {out_png}")
        plt.close()
        return

    df_valid = df[df[key_loss].notna()].copy()
    if df_valid.empty:
        plt.title("Content micro loss (no data)")
        plt.xlabel("Micro index")
        plt.ylabel(key_loss)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[ok] Content 图已保存 -> {out_png} (0 点)")
        return

    if xcol not in df_valid.columns:
        xcol = "row_seq"

    df_valid = df_valid.sort_values(xcol)
    df_valid["plot_x"] = range(1, len(df_valid) + 1)
    df_valid[key_loss] = apply_smoothing(df_valid[key_loss], smooth)

    total_points = len(df_valid)

    for fname, sub in df_valid.groupby("file", sort=False):
        plt.plot(sub["plot_x"], sub[key_loss], label=f"{fname} (n={len(sub)})")

    ttl = "Content micro loss"
    if title_extra:
        ttl += f" | {title_extra}"
    if smooth > 1:
        ttl += f" | MA={smooth}"
    plt.title(ttl)
    plt.xlabel("Micro index (1..N)")
    plt.ylabel(key_loss)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[ok] Content 图已保存 -> {out_png} (共 {total_points} 点)")


def plot_total_micro(df: pd.DataFrame, out_png: str,
                     smooth: int, xcol: str, title_extra: str):
    """
    绘制日志中记录的 train/loss（总 loss）。
    """
    plt.figure(figsize=(10, 5), dpi=150)
    key_loss = "total_loss"

    if key_loss not in df.columns:
        print(f"[warn] 缺少列 {key_loss}，跳过绘制 {out_png}")
        plt.close()
        return

    df_valid = df[df[key_loss].notna()].copy()
    if df_valid.empty:
        plt.title("Total micro loss (no train/loss data)")
        plt.xlabel("Micro index")
        plt.ylabel(key_loss)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[ok] Total 图已保存 -> {out_png} (0 点)")
        return

    if xcol not in df_valid.columns:
        xcol = "row_seq"

    df_valid = df_valid.sort_values(xcol)
    df_valid["plot_x"] = range(1, len(df_valid) + 1)
    df_valid[key_loss] = apply_smoothing(df_valid[key_loss], smooth)

    total_points = len(df_valid)

    for fname, sub in df_valid.groupby("file", sort=False):
        plt.plot(sub["plot_x"], sub[key_loss], label=f"{fname} (n={len(sub)})")

    ttl = "Total micro loss (from train/loss)"
    if title_extra:
        ttl += f" | {title_extra}"
    if smooth > 1:
        ttl += f" | MA={smooth}"
    plt.title(ttl)
    plt.xlabel("Micro index (1..N)")
    plt.ylabel(key_loss)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[ok] Total 图已保存 -> {out_png} (共 {total_points} 点)")


def main():
    ap = argparse.ArgumentParser(
        description="从训练日志解析 micro-batch 级别 loss 并绘图（支持 depth-based head selection）"
    )
    ap.add_argument("-i", "--inputs", nargs="+", required=True, help="一个或多个日志文件路径")
    ap.add_argument("-o", "--out_prefix", default="micro_losses", help="输出文件前缀")
    ap.add_argument("--csv", default=None, help="可选：导出 CSV 路径")
    ap.add_argument("--smooth", type=int, default=1, help="平滑窗口大小，默认1（不平滑）")
    ap.add_argument(
        "--x",
        choices=["row_seq", "micro_global_seq", "micro_idx_in_step"],
        default="row_seq",
        help="内部排序使用的横轴字段（最终都会映射为 1..N）"
    )

    args = ap.parse_args()

    df = concat_with_source(args.inputs)
    if df.empty:
        print("[error] 没有解析到任何 [micro] 数据，检查日志路径与内容。")
        return

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"[ok] 已导出 micro 级 CSV -> {args.csv}")

    title_extra = "micro-level"

    # 绘制四张图
    plot_head_micro(
        df, head_id=0,
        out_png=f"{args.out_prefix}_micro_head0.png",
        smooth=args.smooth,
        xcol=args.x,
        title_extra=title_extra,
    )
    plot_head_micro(
        df, head_id=1,
        out_png=f"{args.out_prefix}_micro_head1.png",
        smooth=args.smooth,
        xcol=args.x,
        title_extra=title_extra,
    )
    plot_content_micro(
        df,
        out_png=f"{args.out_prefix}_micro_content.png",
        smooth=args.smooth,
        xcol=args.x,
        title_extra=title_extra,
    )
    plot_total_micro(
        df,
        out_png=f"{args.out_prefix}_micro_total.png",
        smooth=args.smooth,
        xcol=args.x,
        title_extra=title_extra,
    )


if __name__ == "__main__":
    main()