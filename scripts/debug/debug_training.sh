#!/bin/bash
#
# BFCL Training Debug Script
# Debug the training part using saved rollout data (no SGLang inference)
#
# Usage:
#   1. First run debug_rollout.sh to generate rollout data
#   2. Then run this script: bash scripts/debug_training.sh
#
# This script:
#   1. Only initializes Megatron (no SGLang)
#   2. Loads pre-saved rollout data for fixed input
#   3. Useful for debugging training logic, gradient computation, etc.
#

# Kill existing processes for clean start
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

export PYTHONBUFFERED=16

# Get script directory and project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source model configuration
source "${PROJECT_ROOT}/3rdparty/slime/scripts/models/qwen3-4B-Instruct-2507.sh"

# ============================================================================
# Debug Configuration
# ============================================================================

# Model paths - modify according to your environment
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen3-4B-Instruct-2507}"
REF_LOAD="${REF_LOAD:-/path/to/Qwen3-4B-Instruct-2507_torch_dist}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/checkpoints}"
PROJECT_NAME="${PROJECT_NAME:-t3rl_bfcl}"
LOAD_PATH="${LOAD_PATH:-${CHECKPOINT_DIR}/${PROJECT_NAME}/debug_train}"
SAVE_PATH="${SAVE_PATH:-${CHECKPOINT_DIR}/${PROJECT_NAME}/debug_train}"

# Debug data directory (must match debug_rollout.sh output)
DEBUG_DATA_DIR="${PROJECT_ROOT}/debug_data"

# Check if debug data exists
if [ ! -d "${DEBUG_DATA_DIR}" ]; then
    echo "Error: Debug data directory not found: ${DEBUG_DATA_DIR}"
    echo "Please run debug_rollout.sh first to generate rollout data."
    exit 1
fi

# Use fewer GPUs for debug
NUM_GPUS="${NUM_GPUS:-8}"

# ============================================================================
# Debug Arguments
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   # --load ${LOAD_PATH}                   # Uncomment to resume training from existing checkpoint
   --save ${SAVE_PATH}
   --save-interval 1                      # Save every step for debug
   --megatron-to-hf-mode raw
)

# Smaller batch for debug
ROLLOUT_ARGS=(
   --num-rollout 2
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-response-len 10000
   --global-batch-size 512
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

# Performance args for debug
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
   --entropy-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01                      # Match main script (verl default)
   --adam-beta1 0.9
   --adam-beta2 0.999                       # Match main script (verl default)
   --clip-grad 1.0
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Debug-specific arguments
DEBUG_ARGS=(
   --debug-train-only                     # Only run training, skip rollout
   --load-debug-rollout-data "${DEBUG_DATA_DIR}/rollout_data_{rollout_id}.pt"
   --disable-rollout-global-dataset       # Don't use global dataset (using pre-saved rollout data)
)

# ============================================================================
# Launch Debug Training
# ============================================================================

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
    --temp-dir /tmp/ray_temp_debug

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

echo "============================================"
echo "Starting Training Debug"
echo "Loading rollout data from: ${DEBUG_DATA_DIR}"
echo "============================================"

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
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DEBUG_ARGS[@]}

echo "============================================"
echo "Training Debug Complete"
echo "============================================"
