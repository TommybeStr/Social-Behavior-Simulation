#!/bin/bash

nproc_per_node=8  # 视显卡数量调整
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=/home/zss/Social_Behavior_Simulation/checkpoints/grpo_logs/train_${timestamp}.log


PROJECT_ROOT="/home/zss/Social_Behavior_Simulation"

nohup python -m verl.trainer.main_ppo \
  --config-path ${PROJECT_ROOT}/Config \
  --config-name grpo_config \
  trainer.project_name=social-behavior-grpo \
  trainer.experiment_name=run${nproc_per_node}gpu \
  > ${LOG_FILE} 2>&1 &