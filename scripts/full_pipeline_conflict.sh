#!/bin/bash -l
#
# Conflict-Aware Pipeline: Augment -> Checklist Extraction
#
# This script runs a reduced data processing pipeline in sequence,
# using a single vLLM server instance for efficiency.
#
# Supports distributed processing via NNODES and RANK environment variables.
# Set LOCAL_DATASET=true to disable HuggingFace operations.
#

set -euo pipefail

echo "[INFO] Starting conflict-aware pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=$PRIMUS_OUTPUT_DIR"

# ============================================================================
# Partitioning Configuration (from environment)
# ============================================================================

PARTITION_NUM=${PARTITION_NUM:-${NNODES}}
PARTITION_INDEX=${PARTITION_INDEX:-${RANK}}

echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"

# ============================================================================
# Configuration
# ============================================================================

# Local dataset mode (disable HuggingFace operations)
LOCAL_DATASET=${LOCAL_DATASET:-false}
echo "[INFO] Local dataset mode: $LOCAL_DATASET"

ROOT="/root/code/rlxr_if_data_creation"

# Model configuration
FULL_ANNOTATOR_MODEL="/root/models/Qwen3-235B-A22B-Thinking-2507-FP8"
GPU_NUM=8
GPU_MEM_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM configuration
VLLM_BASE_URL="http://localhost:8000/v1"
VLLM_LOG="$ROOT/vllm_qwen3_235b.log"
TIMEOUT=600

# Processing configuration
N_THREADS=64
MAX_INFLIGHT=128
GENERATION_CONFIG='{"temperature": 0.6, "top_p": 0.95, "extra_body": {"enable_thinking": true, "top_k": 20}}'

# Input dataset and args
INPUT_DATASET=${INPUT_DATASET:-"NOT_SETTED"}
SPLIT=${SPLIT:-"train"}
REPO_NAME_BASE=${REPO_NAME_BASE:-"NOT_SETTED"}

echo "[INFO] Input dataset: $INPUT_DATASET"
echo "[INFO] Repo name base: $REPO_NAME_BASE"

# Augmentation bounds
AUGMENT_LOWER_BOUND=${AUGMENT_LOWER_BOUND:-1}
AUGMENT_UPPER_BOUND=${AUGMENT_UPPER_BOUND:-3}
NUM_CONSTRAINTS=${NUM_CONSTRAINTS:-15}
VERSION=${VERSION:-"conflict_v2"}

# Base repo names
AUGMENT_REPO_NAME_BASE="${REPO_NAME_BASE}_augmented_${SPLIT}_${VERSION}"
CHECKLIST_REPO_NAME_BASE="${REPO_NAME_BASE}_checklist_${SPLIT}_${VERSION}"

# Add partition suffix to output paths if partitioning is enabled
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    PARTITION_SUFFIX="_${PARTITION_INDEX}_of_${PARTITION_NUM}"
    AUGMENT_REPO_NAME="${AUGMENT_REPO_NAME_BASE}${PARTITION_SUFFIX}"
    CHECKLIST_REPO_NAME="${CHECKLIST_REPO_NAME_BASE}${PARTITION_SUFFIX}"
else
    AUGMENT_REPO_NAME="$AUGMENT_REPO_NAME_BASE"
    CHECKLIST_REPO_NAME="$CHECKLIST_REPO_NAME_BASE"
fi

# Step 1: Conflict-Aware Augmentation
AUGMENT_CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$AUGMENT_REPO_NAME"
AUGMENT_OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$AUGMENT_REPO_NAME"
AUGMENT_SYSTEM_PROMPT="$ROOT/prompt/constraint_augmentation_conflict.txt"
AUGMENT_USER_PROMPT="$ROOT/prompt/constraint_augmentation_conflict_user.txt"
AUGMENT_REF_CONSTRAINTS="$ROOT/prompt/reference_constraints_v2.txt"

# Step 2: Checklist Extraction
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
    echo "[STEP 1/2] Running Conflict-Aware Instruction Augmentation"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $AUGMENT_OUTPUT_DIR"
    if [[ "$PARTITION_NUM" -gt 1 ]]; then
        echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"
    fi
    echo ""

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
        --upper_bound "$AUGMENT_UPPER_BOUND"
        --lower_bound "$AUGMENT_LOWER_BOUND"
    )

    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$AUGMENT_REPO_NAME")
    fi

    if [[ "$PARTITION_NUM" -gt 1 ]]; then
        cmd_args+=(--partition_num "$PARTITION_NUM" --partition_index "$PARTITION_INDEX")
    fi

    python "$ROOT/augment_instructions_conflict_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Conflict-aware augmentation complete"
}

run_checklist_extraction() {
    echo ""
    echo "============================================================"
    echo "[STEP 2/2] Running Checklist Extraction"
    echo "============================================================"
    echo "[INFO] Input: $AUGMENT_OUTPUT_DIR"
    echo "[INFO] Output: $CHECKLIST_OUTPUT_DIR"
    echo ""

    local cmd_args=(
        --input_dataset "$AUGMENT_OUTPUT_DIR"
        --save_to_disk "$CHECKLIST_OUTPUT_DIR"
        --model "$FULL_ANNOTATOR_MODEL"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$CHECKLIST_CACHE_DIR"
        --num_workers "$N_THREADS"
        --max_inflight "$MAX_INFLIGHT"
        --system_prompt_path "$CHECKLIST_SYSTEM_PROMPT"
        --user_prompt_path "$CHECKLIST_USER_PROMPT"
        --reference_constraints_path "$CHECKLIST_REF_CONSTRAINTS"
        --instruction_field "augmented_prompt"
        --split "$SPLIT"
        --generation_config "$GENERATION_CONFIG"
    )

    if [[ "$LOCAL_DATASET" != "true" ]]; then
        cmd_args+=(--push_to_hub "JingweiNi/$CHECKLIST_REPO_NAME")
    fi

    python "$ROOT/checklist_extraction_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Checklist extraction complete"
}

copy_results() {
    echo ""
    echo "[INFO] Copying results to persistent storage..."

    cp -r "$AUGMENT_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$CHECKLIST_CACHE_DIR" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true

    echo "[DONE] Results copied"
}

# ============================================================================
# Main
# ============================================================================

if [[ "$LOCAL_DATASET" != "true" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
else
    echo "[INFO] Skipping HuggingFace login (local dataset mode)"
fi

# trap 'stop_vllm' EXIT

start_vllm
wait_for_vllm

run_augmentation
run_checklist_extraction

copy_results

stop_vllm
trap - EXIT

echo ""
echo "============================================================"
echo "[SUCCESS] Conflict-aware pipeline complete!"
echo "============================================================"
if [[ "$LOCAL_DATASET" != "true" ]]; then
    echo "Step 1 - Augmented dataset:  JingweiNi/$AUGMENT_REPO_NAME"
    echo "Step 2 - Checklist dataset:  JingweiNi/$CHECKLIST_REPO_NAME"
else
    echo "Step 1 - Augmented dataset:  $AUGMENT_OUTPUT_DIR"
    echo "Step 2 - Checklist dataset:  $CHECKLIST_OUTPUT_DIR"
fi
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi

echo ""

exit 0
