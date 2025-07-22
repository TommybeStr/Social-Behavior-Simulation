python /home/zss/Social_Behavior_Simulation/data_prepocess/tools/reshape.py /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/Ruby_Face_Cream_r0.jsonl /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split_full.parquet
pkill -u zss -f torchrun

python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/version2.1/make_sft_file.py --input /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/rebuild_data.json --json_output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --parquet_output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split_full.parquet

python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/version2.1/width_split_sft_file.py --input /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/width_split_sft_file.jsonl

python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/compute_len.py --jsonl_path /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --model_path /home/zss/Social_Behavior_Simulation/Qwen2.5-1.5B-Instruct --output_csv /home/zss/Social_Behavior_Simulation/length.csv

python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/filter.py --jsonl  /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --csv /home/zss/Social_Behavior_Simulation/length.csv --output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/filted_split_full.parquet --max-tokens 8192

python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/split_parquet.py --input /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/filted_split_full.parquet --train_output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/train_data.parquet --val_output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/val_data.parquet