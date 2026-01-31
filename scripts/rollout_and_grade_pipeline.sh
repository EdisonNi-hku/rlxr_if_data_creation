#!/bin/bash -l
#
# Rollout and Grade Pipeline
#
# This script:
# 1. Starts vLLM with Model A (rollout model)
# 2. Generates rollouts using rollout_only.py
# 3. Stops Model A, starts Model B (grading model)
# 4. Grades rollouts using analyze_checklist.py with LLM-based checklist evaluation
#
# Supports distributed processing via PARTITION_NUM and PARTITION_INDEX.
#

set -euo pipefail

echo "[INFO] Starting rollout and grade pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-./output}"

# ============================================================================
# Partitioning Configuration (from environment)
# ============================================================================

PARTITION_NUM=${PARTITION_NUM:-${NNODES:-1}}
PARTITION_INDEX=${PARTITION_INDEX:-${RANK:-0}}

echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"

# ============================================================================
# Configuration
# ============================================================================

ROOT=${ROOT:-"/root/code/rlxr_if_data_creation"}
PRIMUS_OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-"$ROOT/output"}

# GPU configuration (shared by both models)
GPU_NUM=${GPU_NUM:-8}
GPU_MEM=${GPU_MEM:-0.9}

# Model A: Rollout model
MODEL_A=${MODEL_A:-"/root/models/Qwen3-8B"}
MODEL_A_EXTRA_ARGS=${MODEL_A_EXTRA_ARGS:-""}

# Model B: Grading model
MODEL_B=${MODEL_B:-"/root/models/Qwen3-235B-A22B-Thinking-2507-FP8"}
MODEL_B_EXTRA_ARGS=${MODEL_B_EXTRA_ARGS:-"--enable-expert-parallel"}

# vLLM configuration
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_LOG_A="$ROOT/vllm_model_a.log"
VLLM_LOG_B="$ROOT/vllm_model_b.log"
TIMEOUT=${TIMEOUT:-600}

# Input configuration
INPUT_DATASET=${INPUT_DATASET:-"NOT_SET"}
CHECKLIST_DATASET=${CHECKLIST_DATASET:-"NOT_SET"}
SPLIT=${SPLIT:-"train"}
INSTRUCTION_FIELD=${INSTRUCTION_FIELD:-"messages"}

# Rollout configuration
NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-4096}
SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH:-""}

# Grading configuration
GRADING_NUM_WORKERS=${GRADING_NUM_WORKERS:-64}
GRADING_MAX_INFLIGHT=${GRADING_MAX_INFLIGHT:-128}
GRADING_GENERATION_CONFIG=${GRADING_GENERATION_CONFIG:-'{"temperature": 0.6, "top_p": 0.95}'}
GRADING_SYSTEM_PROMPT="$ROOT/prompt/checklist_eval.txt"
GRADING_USER_PROMPT="$ROOT/prompt/checklist_eval_user.txt"
GRADING_CACHE_DIR=${GRADING_CACHE_DIR:-"$ROOT/grading_cache"}

# Output configuration
VERSION=${VERSION:-"v1"}
OUTPUT_NAME_BASE=${OUTPUT_NAME_BASE:-"rollouts_${VERSION}"}

# Add partition suffix if partitioning is enabled
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    PARTITION_SUFFIX="_p${PARTITION_INDEX}_of_${PARTITION_NUM}"
else
    PARTITION_SUFFIX=""
fi

# Output paths (rollout_only.py adds partition suffix automatically)
ROLLOUT_OUTPUT_ARG="$PRIMUS_OUTPUT_DIR/${OUTPUT_NAME_BASE}.jsonl"
ROLLOUT_OUTPUT="$PRIMUS_OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}.jsonl"
GRADED_OUTPUT="$PRIMUS_OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}_graded.jsonl"
METRICS_OUTPUT="$PRIMUS_OUTPUT_DIR/${OUTPUT_NAME_BASE}${PARTITION_SUFFIX}_metrics.json"

# Strip thinking tokens
STRIP_THINKING=${STRIP_THINKING:-true}

echo "[INFO] Configuration:"
echo "  Model A (rollout): $MODEL_A"
echo "  Model B (grading): $MODEL_B"
echo "  GPU count: $GPU_NUM"
echo "  Input dataset: $INPUT_DATASET"
echo "  Checklist dataset: $CHECKLIST_DATASET"
echo "  Rollout output: $ROLLOUT_OUTPUT"
echo "  Graded output: $GRADED_OUTPUT"

# ============================================================================
# Functions
# ============================================================================

start_vllm_model_a() {
    echo "[INFO] Starting vLLM server with Model A..."
    local cmd="vllm serve $MODEL_A \
        --port $VLLM_PORT \
        --tensor-parallel-size $GPU_NUM \
        --gpu_memory_utilization $GPU_MEM \
        $MODEL_A_EXTRA_ARGS"

    echo "[INFO] Command: $cmd"
    eval "$cmd" > "$VLLM_LOG_A" 2>&1 &
    VLLM_PID=$!
    echo "[INFO] vLLM Model A PID=$VLLM_PID"
}

start_vllm_model_b() {
    echo "[INFO] Starting vLLM server with Model B..."
    local cmd="vllm serve $MODEL_B \
        --port $VLLM_PORT \
        --tensor-parallel-size $GPU_NUM \
        --gpu_memory_utilization $GPU_MEM \
        $MODEL_B_EXTRA_ARGS"

    echo "[INFO] Command: $cmd"
    eval "$cmd" > "$VLLM_LOG_B" 2>&1 &
    VLLM_PID=$!
    echo "[INFO] vLLM Model B PID=$VLLM_PID"
}

wait_for_vllm() {
    local log_file=$1
    echo "[WAIT] Watching $log_file for readiness..."
    if ! timeout "$TIMEOUT" bash -c "( tail -n0 -f \"$log_file\" & ) | grep -q -- 'Application startup complete.'"; then
        echo "[ERROR] vLLM did not become ready within ${TIMEOUT}s"
        cat "$log_file"
        return 1
    fi
    echo "[READY] vLLM is ready."
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "[CLEANUP] Stopping vLLM (PID=$VLLM_PID)"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        # Give some time for GPU memory to be released
        sleep 5
    fi
}

run_rollout() {
    echo ""
    echo "============================================================"
    echo "[STEP 1/2] Generating Rollouts with Model A"
    echo "============================================================"
    echo "[INFO] Input: $INPUT_DATASET"
    echo "[INFO] Output: $ROLLOUT_OUTPUT"
    echo ""

    # Build command arguments
    local cmd_args=(
        --input_dataset "$INPUT_DATASET"
        --output_path "$ROLLOUT_OUTPUT_ARG"
        --model "$MODEL_A"
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

    # Add system prompt if specified
    if [[ -n "$SYSTEM_PROMPT_PATH" ]]; then
        cmd_args+=(--system_prompt_path "$SYSTEM_PROMPT_PATH")
    fi

    # Add strip_thinking if enabled
    if [[ "$STRIP_THINKING" == "true" ]]; then
        cmd_args+=(--strip_thinking)
    fi

    python "$ROOT/rollout_only.py" "${cmd_args[@]}"

    echo "[DONE] Rollout generation complete"
}

run_grading() {
    echo ""
    echo "============================================================"
    echo "[STEP 2/2] Grading Rollouts with Model B (Checklist Evaluation)"
    echo "============================================================"
    echo "[INFO] Rollouts: $ROLLOUT_OUTPUT"
    echo "[INFO] Checklist: $CHECKLIST_DATASET"
    echo "[INFO] Output: $GRADED_OUTPUT"
    echo ""

    # Build command arguments
    local cmd_args=(
        --input "$ROLLOUT_OUTPUT"
        --grade
        --checklist_dataset "$CHECKLIST_DATASET"
        --split "$SPLIT"
        --model "$MODEL_B"
        --base_url "$VLLM_BASE_URL"
        --cache_path "$GRADING_CACHE_DIR"
        --num_workers "$GRADING_NUM_WORKERS"
        --max_inflight "$GRADING_MAX_INFLIGHT"
        --system_prompt_path "$GRADING_SYSTEM_PROMPT"
        --user_prompt_path "$GRADING_USER_PROMPT"
        --generation_config "$GRADING_GENERATION_CONFIG"
        --save_graded "$GRADED_OUTPUT"
        --output "$METRICS_OUTPUT"
    )

    # Add strip_thinking if enabled
    if [[ "$STRIP_THINKING" == "true" ]]; then
        cmd_args+=(--strip_thinking)
    fi

    python "$ROOT/analyze_checklist.py" "${cmd_args[@]}"

    echo "[DONE] Grading complete"
}

# ============================================================================
# Main
# ============================================================================

# Create output directory
mkdir -p "$PRIMUS_OUTPUT_DIR"

# Check if rollout output already exists (resume support)
if [[ -f "$ROLLOUT_OUTPUT" ]]; then
    echo "[INFO] Rollout output already exists: $ROLLOUT_OUTPUT"
    echo "[INFO] Skipping rollout generation (delete file to regenerate)"
else
    # Step 1: Generate rollouts with Model A
    start_vllm_model_a
    wait_for_vllm "$VLLM_LOG_A"
    run_rollout
    stop_vllm
fi

# Step 2: Grade rollouts with Model B
start_vllm_model_b
wait_for_vllm "$VLLM_LOG_B"
run_grading
stop_vllm

echo ""
echo "============================================================"
echo "[SUCCESS] Pipeline complete!"
echo "============================================================"
echo "Rollouts: $ROLLOUT_OUTPUT"
echo "Graded results: $GRADED_OUTPUT"
echo "Metrics: $METRICS_OUTPUT"
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
