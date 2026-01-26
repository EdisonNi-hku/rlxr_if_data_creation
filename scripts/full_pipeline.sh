#!/bin/bash -l
#
# Full Pipeline: Augment -> Contradiction Check -> Checklist Extraction
#
# This script runs the complete data processing pipeline in sequence,
# using a single vLLM server instance for efficiency.
#
# Supports distributed processing via NNODES and RANK environment variables.
# Set LOCAL_DATASET=true to disable HuggingFace operations.
#

set -euo pipefail

echo "[INFO] Starting full pipeline"
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

# OSS_SAVE_PATH="/primus_datasets/jingwei"
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
GENERATION_CONFIG='{"temperature": 0.6, "top_p": 0.95, "extra_body": {"enable_thinking": true, "top_k": 20}}'

# Input dataset and args
INPUT_DATASET="JingweiNi/magpie_creative_dedup"
NUM_CONSTRAINTS=15
VERSION="v1"

# Base repo names
AUGMENT_REPO_NAME_BASE="magpie_creative_dedup_augmented_${SPLIT}_${VERSION}"
FILTER_REPO_NAME_BASE="magpie_creative_dedup_filtered_${SPLIT}_${VERSION}"
CHECKLIST_REPO_NAME_BASE="magpie_creative_dedup_checklist_${SPLIT}_${VERSION}"

# Add partition suffix to output paths if partitioning is enabled
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    PARTITION_SUFFIX="_${PARTITION_INDEX}_of_${PARTITION_NUM}"
    AUGMENT_REPO_NAME="${AUGMENT_REPO_NAME_BASE}${PARTITION_SUFFIX}"
    FILTER_REPO_NAME="${FILTER_REPO_NAME_BASE}${PARTITION_SUFFIX}"
    CHECKLIST_REPO_NAME="${CHECKLIST_REPO_NAME_BASE}${PARTITION_SUFFIX}"
else
    AUGMENT_REPO_NAME="$AUGMENT_REPO_NAME_BASE"
    FILTER_REPO_NAME="$FILTER_REPO_NAME_BASE"
    CHECKLIST_REPO_NAME="$CHECKLIST_REPO_NAME_BASE"
fi

# Step 1: Augmentation
AUGMENT_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$AUGMENT_REPO_NAME"
AUGMENT_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$AUGMENT_REPO_NAME"
AUGMENT_SYSTEM_PROMPT="$ROOT/prompt/constraint_augmentation.txt"
AUGMENT_USER_PROMPT="$ROOT/prompt/constraint_augmentation_user.txt"
AUGMENT_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

# Step 2: Contradiction Check
FILTER_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$FILTER_REPO_NAME"
FILTER_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$FILTER_REPO_NAME"
FILTER_SYSTEM_PROMPT="$ROOT/prompt/contradiction_check.txt"
FILTER_USER_PROMPT="$ROOT/prompt/constradiction_check_user.txt"
FILTER_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

# Step 3: Checklist Extraction
CHECKLIST_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$CHECKLIST_REPO_NAME"
CHECKLIST_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$CHECKLIST_REPO_NAME"
CHECKLIST_SYSTEM_PROMPT="$ROOT/prompt/checklist_extraction_v1.txt"
CHECKLIST_USER_PROMPT="$ROOT/prompt/checklist_extraction_user.txt"
CHECKLIST_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

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
    echo "[STEP 1/3] Running Instruction Augmentation"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $AUGMENT_OUTPUT_DIR"
    if [[ "$PARTITION_NUM" -gt 1 ]]; then
        echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"
    fi
    echo ""

    # Build command arguments
    local cmd_args=(
        --input_dataset "$INPUT_DATASET"
        --save_to_disk "$AUGMENT_OUTPUT_DIR"
        --model "$FULL_ANNOTATOR_MODEL"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$AUGMENT_CACHE_DIR"
        --num_workers "$N_THREADS"
        --max_inflight "$MAX_INFLIGHT"
        --system_prompt_path "$AUGMENT_SYSTEM_PROMPT"
        --user_prompt_path "$AUGMENT_USER_PROMPT"
        --reference_constraints_path "$AUGMENT_REF_CONSTRAINTS"
        --split "$SPLIT"
        --generation_config "$GENERATION_CONFIG"
        --num_constraints "$NUM_CONSTRAINTS"
    )

    # Add push_to_hub only if not in local dataset mode
    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$AUGMENT_REPO_NAME")
    fi

    # Add partition args only for the first step (partitioning happens here)
    if [[ "$PARTITION_NUM" -gt 1 ]]; then
        cmd_args+=(--partition_num "$PARTITION_NUM" --partition_index "$PARTITION_INDEX")
    fi

    python "$ROOT/augment_instructions_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Augmentation complete"
}

run_contradiction_check() {
    echo ""
    echo "============================================================"
    echo "[STEP 2/3] Running Contradiction Check"
    echo "============================================================"
    echo "[INFO] Input: $AUGMENT_OUTPUT_DIR"
    echo "[INFO] Output: $FILTER_OUTPUT_DIR"
    echo ""

    # Build command arguments
    local cmd_args=(
        --input_dataset "$AUGMENT_OUTPUT_DIR"
        --save_to_disk "$FILTER_OUTPUT_DIR"
        --model "$FULL_ANNOTATOR_MODEL"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$FILTER_CACHE_DIR"
        --num_workers "$N_THREADS"
        --max_inflight "$MAX_INFLIGHT"
        --system_prompt_path "$FILTER_SYSTEM_PROMPT"
        --user_prompt_path "$FILTER_USER_PROMPT"
        --reference_constraints_path "$FILTER_REF_CONSTRAINTS"
        --instruction_field "augmented_prompt"
        --split "$SPLIT"
        --generation_config "$GENERATION_CONFIG"
    )

    # Add push_to_hub only if not in local dataset mode
    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$FILTER_REPO_NAME")
    fi

    # No partition args - input is already partitioned from step 1

    python "$ROOT/contradiction_check_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Contradiction check complete"
}

run_checklist_extraction() {
    echo ""
    echo "============================================================"
    echo "[STEP 3/3] Running Checklist Extraction"
    echo "============================================================"
    echo "[INFO] Input: $FILTER_OUTPUT_DIR"
    echo "[INFO] Output: $CHECKLIST_OUTPUT_DIR"
    echo ""

    # Build command arguments
    local cmd_args=(
        --input_dataset "$FILTER_OUTPUT_DIR"
        --save_to_disk "$CHECKLIST_OUTPUT_DIR"
        --model "$FULL_ANNOTATOR_MODEL"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$CHECKLIST_CACHE_DIR"
        --num_workers "$N_THREADS"
        --max_inflight "$MAX_INFLIGHT"
        --system_prompt_path "$CHECKLIST_SYSTEM_PROMPT"
        --user_prompt_path "$CHECKLIST_USER_PROMPT"
        --reference_constraints_path "$CHECKLIST_REF_CONSTRAINTS"
        --instruction_field "raw_prompt"
        --split "$SPLIT"
        --generation_config "$GENERATION_CONFIG"
    )

    # Add push_to_hub only if not in local dataset mode
    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$CHECKLIST_REPO_NAME")
    fi

    # No partition args - input is already partitioned from step 1

    python "$ROOT/checklist_extraction_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Checklist extraction complete"
}

copy_results() {
    echo ""
    echo "[INFO] Copying results to persistent storage..."
    
    # Copy all cache directories
    cp -r "$AUGMENT_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$FILTER_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$CHECKLIST_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    
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

# Setup cleanup trap (comment out to keep vLLM running after script ends)
# trap 'stop_vllm' EXIT

# Start vLLM server
start_vllm
wait_for_vllm

# Run pipeline
run_augmentation
run_contradiction_check
run_checklist_extraction

# Copy results
copy_results

# Cleanup
stop_vllm
trap - EXIT

echo ""
echo "============================================================"
echo "[SUCCESS] Full pipeline complete!"
echo "============================================================"
if [[ "$LOCAL_DATASET" != "true" ]]; then
    echo "Step 1 - Augmented dataset:  JingweiNi/$AUGMENT_REPO_NAME"
    echo "Step 2 - Filtered dataset:   JingweiNi/$FILTER_REPO_NAME"
    echo "Step 3 - Checklist dataset:  JingweiNi/$CHECKLIST_REPO_NAME"
else
    echo "Step 1 - Augmented dataset:  $AUGMENT_OUTPUT_DIR"
    echo "Step 2 - Filtered dataset:   $FILTER_OUTPUT_DIR"
    echo "Step 3 - Checklist dataset:  $CHECKLIST_OUTPUT_DIR"
fi
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
