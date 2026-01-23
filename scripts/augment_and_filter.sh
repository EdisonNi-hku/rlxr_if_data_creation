#!/bin/bash -l
#
# Chain script: Augment Instructions -> Contradiction Check
#
# This script runs both augmentation and contradiction filtering in sequence,
# using a single vLLM server instance for efficiency.
#

set -euo pipefail

echo "[INFO] Starting augment_and_filter pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=$PRIMUS_OUTPUT_DIR"

# ============================================================================
# Configuration
# ============================================================================

OSS_SAVE_PATH="/primus_datasets/jingwei"
ROOT="/root/code/rlxr_if_data_creation"

# Model configuration
FULL_ANNOTATOR_MODEL="/root/models/Qwen3-235B-A22B-Thinking-2507-FP8"
GPU_NUM=8
GPU_MEM_UTILIZATION=0.8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM configuration
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_LOG="$ROOT/vllm_qwen3_235b.log"
TIMEOUT=600

# Processing configuration
N_THREADS=64
MAX_INFLIGHT=128
SPLIT="train"
_RAW_GENERATION_CONFIG='{"temperature": 0.6, "top_p": 0.95, "extra_body": {"enable_thinking": true, "top_k": 20}}'
GENERATION_CONFIG_ESCAPED=$(printf '%q' "$_RAW_GENERATION_CONFIG")

# Input dataset and args
INPUT_DATASET="JingweiNi/magpie_creative_dedup"
NUM_CONSTRAINTS=15

# Step 1: Augmentation
AUGMENT_REPO_NAME="magpie_creative_dedup_augmented"
AUGMENT_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$AUGMENT_REPO_NAME"
AUGMENT_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$AUGMENT_REPO_NAME"
AUGMENT_SYSTEM_PROMPT="$ROOT/prompt/constraint_augmentation.txt"
AUGMENT_USER_PROMPT="$ROOT/prompt/constraint_augmentation_user.txt"
AUGMENT_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

# Step 2: Contradiction Check
FILTER_REPO_NAME="magpie_creative_dedup_augmented_filtered"
FILTER_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$FILTER_REPO_NAME"
FILTER_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$FILTER_REPO_NAME"
FILTER_SYSTEM_PROMPT="$ROOT/prompt/contradiction_check.txt"
FILTER_USER_PROMPT="$ROOT/prompt/constradiction_check_user.txt"
FILTER_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

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

run_augmentation() {
    echo ""
    echo "============================================================"
    echo "[STEP 1] Running Instruction Augmentation"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $AUGMENT_OUTPUT_DIR"
    echo ""

    python "$ROOT/augment_instructions_vllm.py" \
        --input_dataset "$INPUT_DATASET" \
        --save_to_disk "$AUGMENT_OUTPUT_DIR" \
        --push_to_hub "JingweiNi/$AUGMENT_REPO_NAME" \
        --model "$FULL_ANNOTATOR_MODEL" \
        --base_url "$VLLM_BASE_URL" \
        --cache_path "$AUGMENT_CACHE_DIR" \
        --num_workers "$N_THREADS" \
        --max_inflight "$MAX_INFLIGHT" \
        --system_prompt_path "$AUGMENT_SYSTEM_PROMPT" \
        --user_prompt_path "$AUGMENT_USER_PROMPT" \
        --reference_constraints_path "$AUGMENT_REF_CONSTRAINTS" \
        --split "$SPLIT" \
        --generation_config "$GENERATION_CONFIG_ESCAPED" \
        --num_constraints $NUM_CONSTRAINTS

    echo "[DONE] Augmentation complete"
}

run_contradiction_check() {
    echo ""
    echo "============================================================"
    echo "[STEP 2] Running Contradiction Check"
    echo "============================================================"
    echo "[INFO] Input: $AUGMENT_OUTPUT_DIR"
    echo "[INFO] Output: $FILTER_OUTPUT_DIR"
    echo ""

    python "$ROOT/contradiction_check_vllm.py" \
        --input_dataset "$AUGMENT_OUTPUT_DIR" \
        --save_to_disk "$FILTER_OUTPUT_DIR" \
        --push_to_hub "JingweiNi/$FILTER_REPO_NAME" \
        --model "$FULL_ANNOTATOR_MODEL" \
        --base_url "$VLLM_BASE_URL" \
        --cache_path "$FILTER_CACHE_DIR" \
        --num_workers "$N_THREADS" \
        --max_inflight "$MAX_INFLIGHT" \
        --system_prompt_path "$FILTER_SYSTEM_PROMPT" \
        --user_prompt_path "$FILTER_USER_PROMPT" \
        --reference_constraints_path "$FILTER_REF_CONSTRAINTS" \
        --instruction_field "augmented_prompt" \
        --split "$SPLIT" \
        --generation_config "$GENERATION_CONFIG_ESCAPED"

    echo "[DONE] Contradiction check complete"
}

copy_results() {
    echo ""
    echo "[INFO] Copying results to persistent storage..."
    
    # Copy augmentation results
    cp -r "$AUGMENT_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$AUGMENT_OUTPUT_DIR" "$OSS_SAVE_PATH/" 2>/dev/null || true
    
    # Copy filter results
    cp -r "$FILTER_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$FILTER_OUTPUT_DIR" "$OSS_SAVE_PATH/" 2>/dev/null || true
    
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

# Run pipeline
run_augmentation
run_contradiction_check

# Copy results
copy_results

# Cleanup
stop_vllm
trap - EXIT

echo ""
echo "============================================================"
echo "[SUCCESS] Pipeline complete!"
echo "============================================================"
echo "Augmented dataset: JingweiNi/$AUGMENT_REPO_NAME"
echo "Filtered dataset:  JingweiNi/$FILTER_REPO_NAME"
echo ""

exit 0
