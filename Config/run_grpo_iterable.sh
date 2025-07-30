#!/bin/bash

set -e
set -x

# =================== 参数定义 ===================
MAX_ITERS=5
TARGET_STEPS=32
PROJECT_ROOT="/home/zss/Social_Behavior_Simulation"
DATA_PATH="$PROJECT_ROOT/data_preprocess/scripts/grpo_data"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/grpo_checkpoints"
INIT_MODEL="$PROJECT_ROOT/checkpoints/default/global_step_2079"
REWARD_FUNC="$PROJECT_ROOT/data_preprocess/scripts/tools/grpo_f1_reward.py"
TRAIN_FILE="$DATA_PATH/grpo_train.parquet"
VAL_FILE="$DATA_PATH/grpo_val.parquet"
LOG_DIR="$PROJECT_ROOT/checkpoints/grpo_logs"
N_GPUS=8

mkdir -p "$LOG_DIR"

# =================== 函数：统计样本并计算 batch_size ===================
get_batch_size() {
  num_samples=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))")
  raw_bs=$(( ($num_samples + $TARGET_STEPS - 1) / $TARGET_STEPS ))

  # 向上取整为8的倍数
  aligned_bs=$(( (raw_bs + 7) / 8 * 8 ))
  echo $aligned_bs
}

# =================== 主训练循环 ===================
ckpt_path="$INIT_MODEL"
loss_csv="$LOG_DIR/loss_summary.csv"

# 如果还没创建 loss_summary.csv，就先写表头
if [[ ! -f "$loss_csv" ]]; then
  echo "iter,step,loss" > "$loss_csv"
fi

for ((iter=1; iter<=$MAX_ITERS; iter++)); do
  echo
  echo "[*] 第 $iter 轮训练启动"
  echo "    使用 checkpoint: $ckpt_path"

  iter_tag="iter_$(printf "%04d" $iter)"
  iter_ckpt="$CHECKPOINT_BASE/$iter_tag"
  iter_log="$LOG_DIR/${iter_tag}_log.jsonl"
  mkdir -p "$iter_ckpt"

  # 计算 batch_size
  batch_size=$(get_batch_size)
  echo "[*] 计算 batch_size=$batch_size 以控制 step ≈ $TARGET_STEPS"

  # 执行训练
  nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FUNC" \
    custom_reward_function.name=compute_score \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$ckpt_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.logger='["console"]' \
    trainer.project_name=social-behavior-grpo \
    trainer.experiment_name=$iter_tag \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=3 \
    trainer.val_before_train=true \
    trainer.default_local_dir="$iter_ckpt" \
    trainer.device=cuda \
    critic.strategy=fsdp \
    > "$iter_log" 2>&1

  echo "[*] 第 $iter 轮训练完成，日志保存在 $iter_log"

  # === 提取 loss 写入 CSV ===
  python3 - <<EOF >> "$loss_csv"
import json
log_file = "$iter_log"
with open(log_file, 'r') as f:
    for line in f:
        try:
            obj = json.loads(line)
            step = obj.get("step")
            loss = obj.get("loss") or obj.get("train_loss")
            if step is not None and loss is not None:
                print(f"{iter},{step},{loss}")
        except:
            continue
EOF

  # === 每轮画图保存 ===
  loss_plot="$LOG_DIR/${iter_tag}_loss.png"
  python3 - <<EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("$loss_csv")
df_iter = df[df["iter"] == $iter]

if not df_iter.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(df_iter["step"], df_iter["loss"], marker='o')
    plt.title("Loss Curve - $iter_tag")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("$loss_plot")
EOF

  echo "[*] 已生成第 $iter 轮的 loss 曲线： $loss_plot"

  # === 更新 checkpoint 路径为最新 ===
  ckpt_path="$iter_ckpt/latest"
done

echo
echo "所有 $MAX_ITERS 轮训练已完成，最终 checkpoint 在：$ckpt_path"
