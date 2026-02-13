#!/bin/bash
#
# BFCL Rollout Debug Script
# Debug the inference/rollout part without loading Megatron training components
#
# Usage:
#   bash scripts/debug_rollout.sh
#
# This script:
#   1. Only initializes SGLang (no Megatron)
#   2. Runs rollout on a small subset of data
#   3. Saves rollout results for later training debug
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
LOAD_PATH="${LOAD_PATH:-${CHECKPOINT_DIR}/${PROJECT_NAME}/debug_rollout}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/bfcl/bfcl_train_base.jsonl}"

# Debug output directory
DEBUG_DATA_DIR="${PROJECT_ROOT}/debug_data"
mkdir -p "${DEBUG_DATA_DIR}"

# Use fewer GPUs for debug (can run on single GPU)
NUM_GPUS="${NUM_GPUS:-8}"

# ============================================================================
# Debug Arguments
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${LOAD_PATH}
)

# Smaller rollout for debug
ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 2                       # Small number for debug
   --rollout-batch-size 32                # Match verl train_batch_size
   --n-samples-per-prompt 16              # Match verl rollout.n=16
   --rollout-max-response-len 10000       # Match verl max_response_length for multi-turn
   --rollout-temperature 1.0
   --global-batch-size 512
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

# Performance args for debug (smaller TP for fewer GPUs)
PERF_ARGS=(
   --tensor-model-parallel-size 1         # TP=1 for debug with fewer GPUs
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.80        # Match main script
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
   --custom-config-path ${PROJECT_ROOT}/configs/bfcl/default.yaml
)

# Debug-specific arguments
DEBUG_ARGS=(
   --debug-rollout-only                   # Only run rollout, skip training
   --save-debug-rollout-data "${DEBUG_DATA_DIR}/rollout_data_{rollout_id}.pt"
)

# ============================================================================
# Launch Debug Rollout
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
echo "Starting Rollout Debug"
echo "Debug data will be saved to: ${DEBUG_DATA_DIR}"
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
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${DEBUG_ARGS[@]}

echo "============================================"
echo "Rollout Debug Complete"
echo "Saved rollout data to: ${DEBUG_DATA_DIR}"
echo "============================================"
