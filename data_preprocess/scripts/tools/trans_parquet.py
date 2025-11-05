#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet_to_json_first_10k.py

将 Parquet 转为漂亮的 JSON，但只导出**前一万条**记录。

依赖：
  pip install pandas pyarrow
"""

import pandas as pd
import json

# =================== 用户可修改的部分 ===================
INPUT_FILE   = "/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.29/train.parquet"  # 输入 Parquet 路径
OUTPUT_FILE  = "/home/zss/Social_Behavior_Simulation/check.json"  # 输出 JSON 路径
ORIENT       = "records"   # JSON 结构: records, split, index, columns, values, table
MAX_ROWS     = 10_000      # 仅导出前多少条
# ======================================================

def main():
    # 读取 Parquet
    df = pd.read_parquet(INPUT_FILE)

    # 仅保留前一万条（若不足则全量）
    df_limited = df.head(MAX_ROWS)
    num_records = len(df_limited)

    # 使用 pandas 序列化为 JSON 字符串，确保类型兼容
    json_str = df_limited.to_json(orient=ORIENT, force_ascii=False)

    # 将 JSON 字符串解析为 Python 对象
    data = json.loads(json_str)

    # 写入漂亮的 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"已从 '{INPUT_FILE}' 提取前 {num_records} 条记录并写入 '{OUTPUT_FILE}'。"
        f"（原始共 {len(df)} 条，导出上限 {MAX_ROWS}）"
    )

if __name__ == "__main__":
    main()
