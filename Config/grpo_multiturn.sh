#!/bin/bash
set -Eeuo pipefail
set -x

# ===== 必要：这台机型 P2P 不支持，必须禁用 =====
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1                # 无 IB/RDMA 就关闭，避免误探测
export NCCL_BLOCKING_WAIT=1             # 出错更快暴露，避免长时间沉默
export NCCL_ASYNC_ERROR_HANDLING=1

# ===== 建议：打开诊断，出问题能定位 =====
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
# 可选：固定更稳的算法（有时更稳）
# export NCCL_ALGO=Ring

# ===== 显存分配碎片优化（你原有设置保留）=====
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# ===== 关闭各种 flash 内核（你原有设置保留）=====
export SGLANG_DISABLE_FLASHINFER=1
export FLASHINFER_DISABLE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_FLASH_ATTENTION=0
export PYTORCH_SDPA_FORCE_FLASH=0
export PYTORCH_SDPA_ENABLE_MEM_EFFICIENT_ATTENTION=1

# ===== 仅用 NUMA0 上的 0,1,2,3 四张卡 =====
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4

# ===== 路径与数据（与你一致）=====
PROJECT_ROOT="/home/zss/Social_Behavior_Simulation"
TRAIN_FILE="/home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/train.parquet"
VAL_FILE="/home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/val.parquet"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/grpo_checkpoints"
INIT_MODEL="/home/zss/Social_Behavior_Simulation/checkpoints/default/rebuild3/global_step_2052"
REWARD_FUNC="$PROJECT_ROOT/data_preprocess/scripts/tools/grpo_f1_reward.py"
LOG_DIR="$PROJECT_ROOT/checkpoints/grpo_logs"

mkdir -p "$LOG_DIR" "$CHECKPOINT_BASE"

# ===== 全局 batch（保持不变，必要时可配合梯度累积再调）=====
MAX_BATCH_SIZE=4

# ===== 日志 CSV（与你一致）=====
loss_csv="$LOG_DIR/loss_summary.csv"
[[ -f "$loss_csv" ]] || echo "iter,step,loss,f1_score" > "$loss_csv"

# ===== 本次训练标识 =====
iter=27 #19，还原ray_worker等； 20，修复了trainer中空批跳过问题 21.关闭cpuoffload，修复同步到达问题 23全部重构 24新reward和uid问题 26.newgold，newreward，多层转发@问题，interests重复问题，均修复 27.数据泄露问题修复
iter_tag="iter_$(printf "%.2f" $iter)"
iter_ckpt="$CHECKPOINT_BASE/$iter_tag"
iter_log="$LOG_DIR/${iter_tag}_log.jsonl"
mkdir -p "$iter_ckpt"

echo
echo "[*] 使用自回归多轮 GRPO（BFS 展开）+ 自定义逐步 Reward"
echo "[*] 训练集:     $TRAIN_FILE"
echo "[*] 验证集:     $VAL_FILE"
echo "[*] 初始模型:   $INIT_MODEL"
echo "[*] 输出目录:   $iter_ckpt"
echo "[*] 日志文件:   $iter_log"
echo "[*] batch_size: $MAX_BATCH_SIZE"
echo "[*] GPUs:       $CUDA_VISIBLE_DEVICES (NUMA0)"
echo

# ===== 关键：绑定到 NUMA0，避免跨 NUMA 的 SYS 链路 =====
# numactl 绑定会继承到子进程（包括多进程的各个 rank）
nohup python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="$INIT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.rollout.name=self_play \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$MAX_BATCH_SIZE \
  data.val_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.return_raw_chat=true \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=true \
  algorithm.kl_penalty=kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.1 \
  custom_reward_function.path="$REWARD_FUNC" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.actor.clip_ratio=0.1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$N_GPUS \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=$N_GPUS \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true\
  trainer.logger='["console"]' \
  trainer.project_name=social-behavior-grpo \
  trainer.experiment_name=$iter_tag \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.save_freq=1 \
  trainer.test_freq=0 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.default_local_dir="$iter_ckpt" \
  trainer.device=cuda \
  > "$iter_log" 2>&1

echo "[*] 训练完成，日志保存在: $iter_log"
