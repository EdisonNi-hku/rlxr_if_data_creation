#!/bin/bash -l
#
# Checkpoint Evaluation Pipeline
#
# End-to-end pipeline that:
# 1. Generates 1 rollout per prompt from a HuggingFace dataset using a checkpoint
# 2. Grades responses with both rule-based constraint verification and reward model
#
# Supports distributed processing via PARTITION_NUM and PARTITION_INDEX.
#
# Example usage:
#   # Single-node evaluation
#   MODEL=/path/to/checkpoint \
#   DATASET=JingweiNi/magpie_creative_dedup_verifiable_test_1_5 \
#   OUTPUT_DIR=./eval_results \
#   bash scripts/eval_checkpoint.sh
#
#   # Multi-node (partition 0 of 4)
#   MODEL=/path/to/checkpoint \
#   DATASET=JingweiNi/magpie_creative_dedup_verifiable_test_1_5 \
#   OUTPUT_DIR=./eval_results \
#   PARTITION_NUM=4 PARTITION_INDEX=0 \
#   bash scripts/eval_checkpoint.sh
#
#   # With reward model
#   MODEL=/path/to/checkpoint \
#   DATASET=JingweiNi/magpie_creative_dedup_verifiable_test_1_5 \
#   REWARD_MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B \
#   OUTPUT_DIR=./eval_results \
#   bash scripts/eval_checkpoint.sh

set -euo pipefail

echo "[INFO] Starting checkpoint evaluation pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=${PRIMUS_OUTPUT_DIR}"

# ============================================================================
# Partitioning Configuration
# ============================================================================

PARTITION_NUM=${PARTITION_NUM:-${NNODES:-1}}
PARTITION_INDEX=${PARTITION_INDEX:-${RANK:-0}}

echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"

# ============================================================================
# Configuration
# ============================================================================

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

# Model configuration
MODEL=${MODEL:?"MODEL is required (path to checkpoint)"}
GPU_NUM=${GPU_NUM:-8}
GPU_MEM=${GPU_MEM:-0.9}

# Dataset configuration
DATASET=${DATASET:?"DATASET is required (HuggingFace dataset name or path)"}
SPLIT=${SPLIT:-"train"}
INSTRUCTION_FIELD=${INSTRUCTION_FIELD:-"messages"}

# Rollout configuration
NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-1}
MAX_TOKENS=${MAX_TOKENS:-4096}
SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH:-""}
SAMPLE=${SAMPLE:-""}
STRIP_THINKING=${STRIP_THINKING:-true}

# Grading configuration
COLUMN_NAME=${COLUMN_NAME:-"pass_rate"}
REWARD_MODEL=${REWARD_MODEL:-""}
REWARD_BATCH_SIZE=${REWARD_BATCH_SIZE:-64}
REWARD_DEVICE=${REWARD_DEVICE:-"cuda:0"}
PUSH_TO_HUB=${PUSH_TO_HUB:-""}

# Output configuration
OUTPUT_DIR="$PRIMUS_OUTPUT_DIR"
VERSION=${VERSION:-"v1"}

# Derive a short model name for output file naming
MODEL_BASENAME=$(basename "$MODEL")
OUTPUT_NAME_BASE=${OUTPUT_NAME_BASE:-"eval_${MODEL_BASENAME}_${VERSION}"}

# Add partition suffix
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    PARTITION_SUFFIX="_p${PARTITION_INDEX}_of_${PARTITION_NUM}"
else
    PARTITION_SUFFIX=""
fi

# Output paths
ROLLOUT_OUTPUT_ARG="$OUTPUT_DIR/${OUTPUT_NAME_BASE}.jsonl"
ROLLOUT_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}.jsonl"
ROLLOUT_OUTPUT_PRIMUS="$PRIMUS_OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}.jsonl"
GRADED_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}_graded"
GRADED_JSONL="$OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}_graded.jsonl"

echo "[INFO] Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Split: $SPLIT"
echo "  GPU count: $GPU_NUM"
echo "  Num rollouts: $NUM_ROLLOUTS"
echo "  Temperature: $TEMPERATURE"
echo "  Column name: $COLUMN_NAME"
echo "  Rollout output: $ROLLOUT_OUTPUT"
echo "  Graded output: $GRADED_OUTPUT"
if [[ -n "$REWARD_MODEL" ]]; then
    echo "  Reward model: $REWARD_MODEL"
fi
if [[ -n "$SAMPLE" ]]; then
    echo "  Sample limit: $SAMPLE"
fi

# ============================================================================
# Step 1: Generate rollouts
# ============================================================================

mkdir -p "$OUTPUT_DIR"

if [[ -f "$ROLLOUT_OUTPUT" ]]; then
    echo ""
    echo "[INFO] Rollout output already exists: $ROLLOUT_OUTPUT"
    echo "[INFO] Skipping rollout generation (delete file to regenerate)"
else
    echo ""
    echo "============================================================"
    echo "[STEP 1/2] Generating rollouts"
    echo "============================================================"

    ROLLOUT_ARGS=(
        --input_dataset "$DATASET"
        --output_path "$ROLLOUT_OUTPUT_ARG"
        --model "$MODEL"
        --tensor_parallel_size "$GPU_NUM"
        --gpu_memory_utilization "$GPU_MEM"
        --split "$SPLIT"
        --instruction_field "$INSTRUCTION_FIELD"
        --num_rollouts "$NUM_ROLLOUTS"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --max_tokens "$MAX_TOKENS"
        --trust_remote_code
        --partition_num "$PARTITION_NUM"
        --partition_index "$PARTITION_INDEX"
    )

    if [[ -n "$SYSTEM_PROMPT_PATH" ]]; then
        ROLLOUT_ARGS+=(--system_prompt_path "$SYSTEM_PROMPT_PATH")
    fi
    if [[ -n "$SAMPLE" ]]; then
        ROLLOUT_ARGS+=(--sample "$SAMPLE")
    fi
    if [[ "$STRIP_THINKING" == "true" ]]; then
        ROLLOUT_ARGS+=(--strip_thinking)
    fi

    python "$ROOT/rollout_only.py" "${ROLLOUT_ARGS[@]}"

    echo "[DONE] Rollout generation complete: $ROLLOUT_OUTPUT"
fi

# Always persist rollouts to PRIMUS_OUTPUT_DIR before grading
if [[ "$ROLLOUT_OUTPUT" != "$ROLLOUT_OUTPUT_PRIMUS" ]]; then
    echo "[INFO] Saving rollouts to PRIMUS_OUTPUT_DIR: $ROLLOUT_OUTPUT_PRIMUS"
    mkdir -p "$PRIMUS_OUTPUT_DIR"
    cp -r "$ROLLOUT_OUTPUT" "$ROLLOUT_OUTPUT_PRIMUS" 2>/dev/null || true
fi

# ============================================================================
# Step 2: Grade with constraint verification (+ optional reward model)
# ============================================================================

echo ""
echo "============================================================"
echo "[STEP 2/2] Grading rollouts"
echo "============================================================"

GRADE_ARGS=(
    --input "$ROLLOUT_OUTPUT"
    --input_format rollout
    --dataset "$DATASET"
    --split "$SPLIT"
    --column_name "$COLUMN_NAME"
    --save_local "$GRADED_OUTPUT"
    --save_graded_jsonl "$GRADED_JSONL"
)

if [[ "$STRIP_THINKING" == "true" ]]; then
    GRADE_ARGS+=(--strip_thinking)
else
    GRADE_ARGS+=(--no_strip_thinking)
fi

if [[ -n "$REWARD_MODEL" ]]; then
    # Always shard reward model across all GPUs
    REWARD_DEVICE="auto"
    GRADE_ARGS+=(
        --reward_model "$REWARD_MODEL"
        --reward_batch_size "$REWARD_BATCH_SIZE"
        --reward_device "$REWARD_DEVICE"
    )
fi

if [[ -n "$PUSH_TO_HUB" ]]; then
    GRADE_ARGS+=(--push_to_hub "$PUSH_TO_HUB")
fi

python "$ROOT/grade_external_results.py" "${GRADE_ARGS[@]}"

echo "[DONE] Grading complete"

# ============================================================================
# Copy Results
# ============================================================================

copy_results() {
    if [[ -z "${PRIMUS_OUTPUT_DIR:-}" ]]; then
        return 0
    fi

    if [[ "$OUTPUT_DIR" == "$PRIMUS_OUTPUT_DIR" ]]; then
        return 0
    fi

    echo ""
    echo "[INFO] Copying results to PRIMUS_OUTPUT_DIR..."

    mkdir -p "$PRIMUS_OUTPUT_DIR"
    cp -r "$ROLLOUT_OUTPUT" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$GRADED_OUTPUT" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true
    cp -r "$GRADED_JSONL" "$PRIMUS_OUTPUT_DIR/" 2>/dev/null || true

    echo "[DONE] Results copied"
}

# Copy results (no-op if already in PRIMUS_OUTPUT_DIR)
copy_results

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "[SUCCESS] Evaluation pipeline complete!"
echo "============================================================"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Rollouts: $ROLLOUT_OUTPUT"
echo "  Graded dataset: $GRADED_OUTPUT"
echo "  Graded JSONL: $GRADED_JSONL"
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "  Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
