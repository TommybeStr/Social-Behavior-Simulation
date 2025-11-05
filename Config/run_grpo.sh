#!/bin/bash

set -x

# =================== 基本参数 ===================
nproc_per_node=8  # 根据 GPU 数量调整
timestamp=$(date +"%Y%m%d_%H%M%S")
PROJECT_ROOT="/home/zss/Social_Behavior_Simulation"
LOG_FILE="${PROJECT_ROOT}/checkpoints/grpo_logs/train_${timestamp}.log"

# 模型路径（ckpt + tokenizer config）
LLM="${PROJECT_ROOT}/checkpoints/default/global_step_2079"
DIST_CKPT_PATH="$LLM"

# 数据路径
train_files="${PROJECT_ROOT}/data_preprocess/scripts/grpo_data/grpo_train.parquet"
test_files="${PROJECT_ROOT}/data_preprocess/scripts/grpo_data/grpo_val.parquet"

# 训练结果路径
CKPT_DIR="${PROJECT_ROOT}/checkpoints/grpo_checkpoints"

# =================== OFFLOAD 设置 ===================
ALL_OFFLOAD=True
ACTOR_PARAM_OFFLOAD=$ALL_OFFLOAD
ACTOR_GRAD_OFFLOAD=$ALL_OFFLOAD
ACTOR_OPTIMIZER_OFFLOAD=$ALL_OFFLOAD
REF_PARAM_OFFLOAD=$ALL_OFFLOAD
CRITIC_PARAM_OFFLOAD=$ALL_OFFLOAD
CRITIC_GRAD_OFFLOAD=$ALL_OFFLOAD
CRITIC_OPTIMIZER_OFFLOAD=$ALL_OFFLOAD

# =================== 分布式参数 ===================
NODES=1
PP=1
TP=1
EP=1
ETP=1
INFER_TP=1
n_resp_per_prompt=4

# =================== 启动训练 ===================
nohup python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  custom_reward_function.path="${PROJECT_ROOT}/data_preprocess/scripts/tools/grpo_f1_reward.py" \
  custom_reward_function.name=compute_score \
  data.train_files="$train_files" \
  data.val_files="$test_files" \
  data.train_batch_size=8 \
  data.val_batch_size=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=true \
  data.truncation=right \
  actor_rollout_ref.model.path="$LLM" \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.load_format=dummy_megatron \
  actor_rollout_ref.rollout.multi_turn.enable=false \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.hybrid_engine=true \
  trainer.logger='["console"]' \
  trainer.project_name=social-behavior-grpo \
  trainer.experiment_name=run${nproc_per_node}gpu \
  trainer.nnodes=$NODES \
  trainer.n_gpus_per_node=$nproc_per_node \
  trainer.save_freq=1000 \
  trainer.test_freq=1000 \
  trainer.total_epochs=3 \
  trainer.val_before_train=true \
  trainer.default_local_dir=$CKPT_DIR \
  trainer.device=cuda \
  critic.strategy=fsdp \
  > ${LOG_FILE} 2>&1 &
  "$@"