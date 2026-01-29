#!/bin/bash
# Multi-Turn FC RL Training Script
# Make sure your current working directory is the root of the project (Agent-RL)

set -x

ulimit -n 65535

now() {
    date '+%Y-%m-%d_%H-%M-%S'
}
TIMESTAMP=$(now)
PROJECT_DIR="$(pwd)"

export TRITON_CACHE_DIR="/tmp/triton_cache_$(whoami)"
export PYTHONPATH="$PROJECT_DIR/verl:$PYTHONPATH"
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
LOG_DIR="logs"
mkdir -p $LOG_DIR

HOME="/ossfs/workspace/"
USER_ROOT_PATH="/agent/share/public/*"

# Project configuration
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/env_tuning/config"
PROJECT_NAME="bfcl_multi_turn_grpo"
EXPERIMENT_NAME="Qwen2.5-7B-Instruct-format-stage1-new-env-enhanced_test_new_code_bfcl_multi_turn_rl_$(now)"
MODEL="/agent/*Your local model path."
DATA_DIR="$PROJECT_DIR/data"
ROLLOUT_DIR="$USER_ROOT_PATH/rollout/$PROJECT_NAME/$EXPERIMENT_NAME"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/bfcl_multi_turn_rl_train_${TIMESTAMP}.log"
DEFAULT_LOCAL_DIR="$USER_ROOT_PATH/models/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"

TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=32
MAX_TOEKN_LEN_PER_GPU=65536
EPOCH=20

export TENSORBOARD_DIR="$HOME/tensorboard/$EXPERIMENT_NAME" 


# Run training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multi_turn_fc_grpo_stage1' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files=$DATA_DIR/bfcl_train_base.parquet \
    data.val_files=$DATA_DIR/bfcl_val.parquet  \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOEKN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOEKN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOEKN_LEN_PER_GPU \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR\
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=$EPOCH 2>&1 | tee $LOG_FILE