python /home/zss/Social_Behavior_Simulation/data_prepocess/tools/reshape.py /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_raw/Ruby_Face_Cream_r0.jsonl /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split_full.parquet
pkill -u zss -f python

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/version2.1/make_sft_file.py --input /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_raw/rebuild2.json  --output_dir /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.28

#python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/version2.1/width_split_sft_file.py --input /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/width_split_sft_file.jsonl

#python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/compute_len.py --jsonl_path /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --model_path /home/zss/Social_Behavior_Simulation/Qwen2.5-1.5B-Instruct --output_csv /home/zss/Social_Behavior_Simulation/length.csv

#python /home/zss/Social_Behavior_Simulation/data_prepocess/scripts/tools/filter.py --jsonl  /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/split.jsonl --csv /home/zss/Social_Behavior_Simulation/length.csv --output /home/zss/Social_Behavior_Simulation/data_prepocess/sft_data_split/filted_split_full.parquet --max-tokens 8192

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/split_parquet.py --input /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/filted_split_full.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/train_data.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/val_data.parquet


#grpo

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/compute_len_grpo.py --parquet_path /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train.parquet --model_path /home/zss/Social_Behavior_Simulation/checkpoints/default/global_step_2079 --output_csv /home/zss/Social_Behavior_Simulation/grpo_length.csv

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/convert_sft_to_grpo.py --sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/split_full.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_val.parquet

#python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/split_parquet.py --input /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/grpo_data/grpo_system.parquet --train_output  /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/grpo_data/train_system.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/grpo_data/val_system.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/paint_loss.py --log /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/iter_0020_log.jsonl --csv /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/loss_summary.csv --plot /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/loss_iter20.png

#python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/convert_to_grpo_multiturn.py --sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/split_full.parquet --grpo_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_grpo.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/split_parquet.py --input /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_grpo.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_val.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/grpo_check.py --grpo_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/train_cleaned.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/val_cleaned.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/version2.1/make_grpo.py --train_sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.28/train.parquet --val_sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.28/val.parquet --test_sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/renew_10.28/test.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/train.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/val.parquet --test_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/test.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/grpo_filter.py --input /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train.parquet --output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_train_filted.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/multiturn_evaluate.py --data /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/multiturn_val.parquet --model /home/zss/Social_Behavior_Simulation/checkpoints/grpo_checkpoints/ite12_globalstep100 --log_path /home/zss/Social_Behavior_Simulation/evaluatelogs.jsonl

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/version2.1/compute_space.py --sft_parquet /home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_split/split_full.parquet --train_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/train_cleaned.parquet --val_output /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/val_cleaned.parquet

python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/paint_reward.py --log /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/iter_26.00_log.jsonl --csv /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/loss_summary.csv --plot /home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/loss_iter26.png

nohup python /home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/model_evaluate.py --data /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/val_min.parquet    --model /home/zss/Social_Behavior_Simulation/checkpoints/default/renew_10.29_2/global_step_800    --tokenizer /home/zss/Social_Behavior_Simulation/checkpoints/default/renew_10.29_2/tokenizer_with_spans --jsonl_overview /home/zss/Social_Behavior_Simulation/checkpoints/evaluate/rl_evaluate.jsonl --jsonl_detail /home/zss/Social_Behavior_Simulation/checkpoints/evaluate/rl_evaluate_detail.jsonl --report_path /home/zss/Social_Behavior_Simulation/checkpoints/evaluate/report.txt --undirected_graph --cls_head0 /home/zss/Social_Behavior_Simulation/checkpoints/default/renew_10.29_2/global_step_800/cls_head0.pt --cls_head_ge1 /home/zss/Social_Behavior_Simulation/checkpoints/default/renew_10.29_2/global_step_800/cls_head1.pt