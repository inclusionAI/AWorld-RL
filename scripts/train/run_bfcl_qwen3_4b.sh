#!/bin/bash
#
# BFCL Multi-Turn Function Calling Training Script
# Based on slime framework with Qwen3-4B model
#
# Hardware: 8x H200 (single node)
#
# Usage:
#   bash scripts/train/run_bfcl_qwen3_4b.sh
#
# Prerequisites:
#   1. Prepare data: python data/preprocess_bfcl_data.py
#   2. Convert model weights to slime format (see slime documentation)
#

# Kill existing processes for clean start
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Get script directory and project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source model configuration
source "${PROJECT_ROOT}/3rdparty/slime/scripts/models/qwen3-4B-Instruct-2507.sh"

# ============================================================================
# Configuration - Modify these paths according to your environment
# ============================================================================

# Model paths (input)
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen3-4B-Instruct-2507}"
REF_LOAD="${REF_LOAD:-/path/to/Qwen3-4B-Instruct-2507_torch_dist}"

# Base directories
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
ROLLOUT_BASE_DIR="${ROLLOUT_BASE_DIR:-${PROJECT_ROOT}/outputs/rollout}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/bfcl/bfcl_train_base.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/bfcl/bfcl_val.jsonl}"

# Hardware configuration: 8x H200 single node
NUM_GPUS="${NUM_GPUS:-8}"

# ============================================================================
# Experiment Configuration
# ============================================================================

PROJECT_NAME="${PROJECT_NAME:-t3rl_bfcl}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"

# Support resuming from existing experiment
# Usage: RESUME_EXPERIMENT=Qwen3-4B-Instruct-2507_bfcl_20260204_091000 bash scripts/train/run_bfcl_qwen3_4b.sh
#
# Resume policy for reliable step continuity:
# - Always use raw mode
# - Initial run: do not pass --load (slime will load from --ref-load)
# - Resume run: pass --load <experiment checkpoint dir>
LOAD_PATH=""

if [ -n "${RESUME_EXPERIMENT:-}" ]; then
    EXPERIMENT_NAME="${RESUME_EXPERIMENT}"
    EXPECTED_CKPT_DIR="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

    # Validate checkpoint directory exists
    if [ ! -d "${EXPECTED_CKPT_DIR}" ]; then
        echo "ERROR: Checkpoint directory not found: ${EXPECTED_CKPT_DIR}"
        echo "Available experiments:"
        ls -la "${CHECKPOINT_DIR}/${PROJECT_NAME}/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    TRACKER_FILE="${EXPECTED_CKPT_DIR}/latest_checkpointed_iteration.txt"
    if [ ! -f "${TRACKER_FILE}" ]; then
        echo "ERROR: Cannot resume reliably without ${TRACKER_FILE}"
        echo "Please verify checkpoint integrity before resuming."
        exit 1
    fi

    LOAD_PATH="${EXPECTED_CKPT_DIR}"

    echo "Resuming experiment: ${EXPERIMENT_NAME}"
    echo "Checkpoint mode: raw"
    echo "Load path: ${LOAD_PATH}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_NAME="${MODEL_NAME}_bfcl_${TIMESTAMP}"
    echo "Starting new experiment: ${EXPERIMENT_NAME}"
    echo "Checkpoint mode: raw"
fi

# Output paths (all use same experiment name for traceability)
SAVE_PATH="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
TENSORBOARD_DIR="${LOG_DIR}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DIR="${ROLLOUT_BASE_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
CONSOLE_LOG_DIR="${LOG_DIR}/console/${PROJECT_NAME}"
CONSOLE_LOG="${CONSOLE_LOG_DIR}/${EXPERIMENT_NAME}.log"
# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${TENSORBOARD_DIR}"
mkdir -p "${ROLLOUT_DIR}"
mkdir -p "${SAVE_PATH}"
mkdir -p "${CONSOLE_LOG_DIR}"

export TENSORBOARD_DIR

# TensorBoard best practice for resume:
# reuse the same TENSORBOARD_DIR so curves continue from the resumed rollout_id.

echo "============================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Save Path: ${SAVE_PATH}"
echo "TensorBoard Dir: ${TENSORBOARD_DIR}"
echo "Rollout Dir: ${ROLLOUT_DIR}"
echo "Console Log: ${CONSOLE_LOG}"
echo "============================================"

# Log experiment metadata to console log
{
    echo "============================================"
    echo "Start Time: $(date)"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Model: ${HF_CHECKPOINT}"
    echo "Train Data: ${TRAIN_DATA}"
    echo "Val Data: ${VAL_DATA}"
    echo "Save Path: ${SAVE_PATH}"
    echo "============================================"
} >> "${CONSOLE_LOG}"

# ============================================================================
# Training Arguments
# ============================================================================

# Checkpoint arguments
# Always use raw mode for both initial run and resume.
CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --save ${SAVE_PATH}
   --save-interval 20
   --megatron-to-hf-mode raw
)

# Add --load only when resuming.
if [ -n "${RESUME_EXPERIMENT:-}" ]; then
   CKPT_ARGS+=(--load ${LOAD_PATH})
fi

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 800
   --rollout-batch-size 32                # Match verl train_batch_size
   --n-samples-per-prompt 16              # Match verl rollout.n=16
   --rollout-max-response-len 10000       # Match verl max_response_length for multi-turn
   --rollout-temperature 1.0
   --global-batch-size 512
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data bfcl ${VAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 20000          # Match rollout for multi-turn
)

# Performance args for 8x H200
# Memory optimization: Use TP=2 and lower max-tokens-per-gpu for long multi-turn sequences
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --log-probs-chunk-size 4096              # Match main script
   --recompute-loss-function                # Match main script
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.001                   # Match verl entropy_coeff
   --eps-clip 0.2                         # Match verl clip_ratio_low
   --eps-clip-high 0.28                   # Match verl clip_ratio_high
   --get-mismatch-metrics
   --custom-tis-function-path t3rl.mismatch.mis.compute_mis_weights_with_cp
)

# Optimizer arguments
# Note: verl defaults vs slime defaults comparison:
#   - lr: verl=1e-6 (overridden), slime=1e-6 (same)
#   - weight_decay: verl=0.01, slime=0.1 (using slime default)
#   - adam_beta2: verl=0.999, slime=0.98 (using slime default)
#   - clip_grad: verl=1.0, slime uses --clip-grad (default 1.0)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                                # Match verl actor.optim.lr
   --lr-decay-style constant                # Match verl lr_scheduler_type
   --weight-decay 0.01                      # Match verl weight_decay (slime default is 0.1)
   --adam-beta1 0.9                         # Match verl betas[0]
   --adam-beta2 0.999                       # Match verl betas[1] (slime default is 0.98)
   --clip-grad 1.0                          # Match verl clip_grad
)

# TensorBoard logging
TENSORBOARD_ARGS=(
   --use-tensorboard
)

# WandB logging (optional, uncomment to enable)
WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-bfcl
   # --wandb-group qwen3-4B
   # --wandb-key ${WANDB_KEY}
)

# SGLang configuration for 8 GPUs (H200 has 140GB memory)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.80      # Match verl gpu_memory_utilization=0.8
   --sglang-attention-backend fa3
   --sglang-chunked-prefill-size 4096
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# BFCL custom generate function
CUSTOM_ARGS=(
   --custom-generate-function-path t3rl.rollout.bfcl.generate
   --custom-rollout-log-function-path t3rl.rollout.bfcl.log_rollout_data
   --custom-eval-rollout-log-function-path t3rl.eval.bfcl.log_eval_rollout_data
   --eval-function-path t3rl.rollout.bfcl.eval_rollout_with_metrics
   --custom-config-path ${PROJECT_ROOT}/configs/bfcl/default.yaml
   --dump-details ${ROLLOUT_DIR}            # Save rollout/train data for analysis
)

# ============================================================================
# Launch Training
# ============================================================================

# Launch ray head node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MEGATRON_ROOT="${MEGATRON_ROOT:-}"

if [ -n "${MEGATRON_ROOT}" ]; then
  RUNTIME_PYTHONPATH="${MEGATRON_ROOT}:${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/slime"
else
  RUNTIME_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/3rdparty/slime"
fi

ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${NUM_GPUS} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir /tmp/ray_temp

# Set up runtime environment
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${RUNTIME_PYTHONPATH}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TENSORBOARD_DIR": "${TENSORBOARD_DIR}"
  }
}
EOF
)

# Submit training job (output to both terminal and log file)
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_ROOT}/3rdparty/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee -a "${CONSOLE_LOG}"

# Log end time
echo "============================================" | tee -a "${CONSOLE_LOG}"
echo "End Time: $(date)" | tee -a "${CONSOLE_LOG}"
echo "============================================" | tee -a "${CONSOLE_LOG}"
