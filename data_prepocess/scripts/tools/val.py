import pandas as pd
import json

# 修复 train.parquet
df_train = pd.read_parquet('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/train_data.parquet')
df_train['messages'] = df_train['messages'].apply(json.loads)
df_train.to_parquet('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/train_fixed.parquet', index=False)
print(f"train.parquet 修复完成，共 {len(df_train)} 条记录")

# 修复 val.parquet
df_val = pd.read_parquet('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/val_data.parquet')
df_val['messages'] = df_val['messages'].apply(json.loads)
df_val.to_parquet('/home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/val_fixed.parquet', index=False)
print(f"val.parquet 修复完成，共 {len(df_val)} 条记录")