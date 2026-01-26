#!/bin/bash -l
#
# Checklist Extraction Pipeline
#
# This script extracts verification checklists from instructions
# using a vLLM server for generation.
#
# Supports distributed processing via NNODES and RANK environment variables.
# Set LOCAL_DATASET=true to disable HuggingFace operations.
#

set -euo pipefail

echo "[INFO] Starting checklist extraction pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=$PRIMUS_OUTPUT_DIR"

# ============================================================================
# Partitioning Configuration (from environment)
# ============================================================================

PARTITION_NUM=${NNODES:-1}
PARTITION_INDEX=${RANK:-0}

echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"

# ============================================================================
# Configuration
# ============================================================================

# Local dataset mode (disable HuggingFace operations)
LOCAL_DATASET=${LOCAL_DATASET:-false}
echo "[INFO] Local dataset mode: $LOCAL_DATASET"

OSS_SAVE_PATH="/primus_datasets/jingwei"
ROOT="/root/code/rlxr_if_data_creation"

# Model configuration
FULL_ANNOTATOR_MODEL="/root/models/Qwen3-235B-A22B-Instruct-2507-FP8"
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
INPUT_DATASET=${INPUT_DATASET:-"NOT_SETTED"}
REPO_NAME_BASE=${REPO_NAME_BASE:-"NOT_SETTED"}

echo "[INFO] Input dataset: $INPUT_DATASET"
echo "[INFO] Repo name base: $REPO_NAME_BASE"

# Add partition suffix to output paths if partitioning is enabled
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    REPO_NAME="${REPO_NAME_BASE}_${PARTITION_INDEX}_of_${PARTITION_NUM}"
else
    REPO_NAME="$REPO_NAME_BASE"
fi

CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$REPO_NAME"
OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$REPO_NAME"

# Prompt paths
SYSTEM_PROMPT_PATH="$ROOT/prompt/checklist_extraction_v1.txt"
USER_PROMPT_PATH="$ROOT/prompt/checklist_extraction_user.txt"

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

run_checklist_extraction() {
    echo ""
    echo "============================================================"
    echo "[STEP] Running Checklist Extraction"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $OUTPUT_DIR"
    echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"
    echo ""

    # Build command arguments
    local cmd_args=(
        --input_dataset "$INPUT_DATASET"
        --save_to_disk "$OUTPUT_DIR"
        --model "$FULL_ANNOTATOR_MODEL"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$CACHE_DIR"
        --num_workers "$N_THREADS"
        --max_inflight "$MAX_INFLIGHT"
        --system_prompt_path "$SYSTEM_PROMPT_PATH"
        --user_prompt_path "$USER_PROMPT_PATH"
        --split "$SPLIT"
        --generation_config "$GENERATION_CONFIG"
        --partition_num "$PARTITION_NUM"
        --partition_index "$PARTITION_INDEX"
    )

    # Add push_to_hub only if not in local dataset mode
    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$REPO_NAME")
    fi

    python "$ROOT/checklist_extraction_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Checklist extraction complete"
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

# Login to HuggingFace (skip if local dataset mode)
if [[ "$LOCAL_DATASET" != "true" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
else
    echo "[INFO] Skipping HuggingFace login (local dataset mode)"
fi

# Setup cleanup trap
trap 'stop_vllm' EXIT

# Start vLLM server
start_vllm
wait_for_vllm

# Run checklist extraction
run_checklist_extraction

# Copy results
copy_results

# Cleanup
stop_vllm
trap - EXIT

echo ""
echo "============================================================"
echo "[SUCCESS] Checklist extraction pipeline complete!"
echo "============================================================"
if [[ "$LOCAL_DATASET" != "true" ]]; then
    echo "Checklist dataset: JingweiNi/$REPO_NAME"
else
    echo "Checklist dataset (local): $OUTPUT_DIR"
fi
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
