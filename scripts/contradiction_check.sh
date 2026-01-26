#!/bin/bash -l
#
# Contradiction Check Pipeline
#
# This script runs contradiction checking on augmented instruction data,
# filtering out instructions with contradictory constraints.
#

set -euo pipefail

echo "[INFO] Starting contradiction check pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=$PRIMUS_OUTPUT_DIR"

# ============================================================================
# Configuration
# ============================================================================

OSS_SAVE_PATH="/primus_datasets/jingwei"
ROOT="/root/code/rlxr_if_data_creation"

# Model configuration
FULL_ANNOTATOR_MODEL="/root/models/Qwen3-235B-A22B-Thinking-2507-FP8"
GPU_NUM=8
GPU_MEM_UTILIZATION=0.7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM configuration
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_LOG="$ROOT/vllm_qwen3_235b.log"
TIMEOUT=600

# Processing configuration
N_THREADS=64
MAX_INFLIGHT=128
SPLIT="train"
DEBUG=0
GENERATION_CONFIG='{"temperature": 0.6, "top_p": 0.95, "extra_body": {"enable_thinking": true, "top_k": 20}}'

# Dataset configuration
INPUT_DATASET="JingweiNi/magpie_creative_dedup_verifiable_if"
REPO_NAME="magpie_creative_dedup_verifiable_if_filtered"
CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$REPO_NAME"
OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$REPO_NAME"

# Prompt paths
SYSTEM_PROMPT_PATH="$ROOT/prompt/contradiction_check.txt"
USER_PROMPT_PATH="$ROOT/prompt/constradiction_check_user.txt"

# ============================================================================
# Functions
# ============================================================================

start_vllm() {
    echo "[INFO] Starting vLLM server..."
    vllm serve "$FULL_ANNOTATOR_MODEL" \
        --tensor-parallel-size "$GPU_NUM" \
        --enable-expert-parallel \
        --gpu_memory_utilization "$GPU_MEM_UTILIZATION" \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "[INFO] vLLM PID=$VLLM_PID"
}

wait_for_vllm() {
    echo "[WAIT] Watching $VLLM_LOG for readiness..."
    if ! timeout "$TIMEOUT" bash -c "( tail -n0 -f \"$VLLM_LOG\" & ) | grep -q -- 'Application startup complete.'"; then
        echo "[ERROR] vLLM did not become ready within ${TIMEOUT}s"
        return 1
    fi
    echo "[READY] vLLM is ready."
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "[CLEANUP] Stopping vLLM ($VLLM_PID)"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

run_contradiction_check() {
    echo ""
    echo "============================================================"
    echo "[STEP] Running Contradiction Check"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $OUTPUT_DIR"
    echo ""

    python "$ROOT/contradiction_check_vllm.py" \
        --input_dataset "$INPUT_DATASET" \
        --save_to_disk "$OUTPUT_DIR" \
        --push_to_hub "JingweiNi/$REPO_NAME" \
        --model "$FULL_ANNOTATOR_MODEL" \
        --base_url "$VLLM_BASE_URL" \
        --cache_path "$CACHE_DIR" \
        --num_workers "$N_THREADS" \
        --max_inflight "$MAX_INFLIGHT" \
        --system_prompt_path "$SYSTEM_PROMPT_PATH" \
        --user_prompt_path "$USER_PROMPT_PATH" \
        --split "$SPLIT" \
        --generation_config "$GENERATION_CONFIG"

    echo "[DONE] Contradiction check complete"
}

copy_results() {
    echo ""
    echo "[INFO] Copying results to persistent storage..."
    
    cp -r "$CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$OUTPUT_DIR" "$OSS_SAVE_PATH/" 2>/dev/null || true
    
    echo "[DONE] Results copied"
}

# ============================================================================
# Main
# ============================================================================

# Login to HuggingFace
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# Setup cleanup trap
trap 'stop_vllm' EXIT

# Start vLLM server
start_vllm
wait_for_vllm

# Run contradiction check
run_contradiction_check

# Copy results
copy_results

# Cleanup
stop_vllm
trap - EXIT

echo ""
echo "============================================================"
echo "[SUCCESS] Contradiction check pipeline complete!"
echo "============================================================"
echo "Filtered dataset: JingweiNi/$REPO_NAME"
echo ""

exit 0
