#!/bin/bash -l
#
# Grade Rollouts Only
#
# This script:
# 1. Starts vLLM with Model B (grading model)
# 2. Grades rollouts using analyze_checklist.py with LLM-based checklist evaluation
#
# Supports distributed processing via PARTITION_NUM and PARTITION_INDEX.
#

set -euo pipefail

echo "[INFO] Starting grading-only pipeline"
echo "[INFO] PRIMUS_OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-}"

# ============================================================================
# Partitioning Configuration (from environment)
# ============================================================================

PARTITION_NUM=${PARTITION_NUM:-${NNODES:-1}}
PARTITION_INDEX=${PARTITION_INDEX:-${RANK:-0}}

echo "[INFO] Partition: $PARTITION_INDEX of $PARTITION_NUM"

# ============================================================================
# Configuration
# ============================================================================

ROOT=${ROOT:-"$(pwd)"}

# GPU configuration (grading model)
GPU_NUM=${GPU_NUM:-8}
GPU_MEM_B=${GPU_MEM_B:-0.8}

# Model B: Grading model
MODEL_B=${MODEL_B:-"/root/models/Qwen3-235B-A22B-Thinking-2507-FP8"}
MODEL_B_EXTRA_ARGS=${MODEL_B_EXTRA_ARGS:-"--enable-expert-parallel"}

# vLLM configuration
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_LOG_B="$ROOT/vllm_model_b.log"
TIMEOUT=${TIMEOUT:-600}

# Input configuration
ROLLOUT_INPUT=${1:-${ROLLOUT_INPUT:-""}}
CHECKLIST_DATASET=${CHECKLIST_DATASET:-"NOT_SET"}
SPLIT=${SPLIT:-"train"}

# Grading configuration
GRADING_NUM_WORKERS=${GRADING_NUM_WORKERS:-64}
GRADING_MAX_INFLIGHT=${GRADING_MAX_INFLIGHT:-128}
GRADING_GENERATION_CONFIG=${GRADING_GENERATION_CONFIG:-'{"temperature": 0.6, "top_p": 0.95}'}
GRADING_SYSTEM_PROMPT="${GRADING_SYSTEM_PROMPT:-"$ROOT/prompt/checklist_eval.txt"}"
GRADING_USER_PROMPT="${GRADING_USER_PROMPT:-"$ROOT/prompt/checklist_eval_user.txt"}"
GRADING_CACHE_DIR=${GRADING_CACHE_DIR:-"$ROOT/grading_cache"}

# Output configuration
GRADED_OUTPUT=${GRADED_OUTPUT:-""}
METRICS_OUTPUT=${METRICS_OUTPUT:-""}

# Debugging
SAMPLE=${SAMPLE:-""}

# Strip thinking tokens
STRIP_THINKING=${STRIP_THINKING:-true}

if [[ -z "$ROLLOUT_INPUT" ]]; then
    echo "[ERROR] Rollout input not set. Provide as first arg or set ROLLOUT_INPUT."
    exit 1
fi
if [[ "$CHECKLIST_DATASET" == "NOT_SET" ]]; then
    echo "[ERROR] CHECKLIST_DATASET not set."
    exit 1
fi

if [[ -z "$GRADED_OUTPUT" ]]; then
    GRADED_OUTPUT="${ROLLOUT_INPUT%.jsonl}_graded.jsonl"
fi
if [[ -z "$METRICS_OUTPUT" ]]; then
    METRICS_OUTPUT="${ROLLOUT_INPUT%.jsonl}_metrics.json"
fi

echo "[INFO] Configuration:"
echo "  Model B (grading): $MODEL_B"
echo "  GPU count: $GPU_NUM"
echo "  Rollout input: $ROLLOUT_INPUT"
echo "  Checklist dataset: $CHECKLIST_DATASET"
echo "  Graded output: $GRADED_OUTPUT"
echo "  Metrics output: $METRICS_OUTPUT"
if [[ -n "$SAMPLE" ]]; then
    echo "  Sample limit: $SAMPLE"
fi

# ============================================================================
# Functions
# ============================================================================

start_vllm_model_b() {
    echo "[INFO] Starting vLLM server with Model B..."
    local cmd="vllm serve $MODEL_B \
        --port $VLLM_PORT \
        --tensor-parallel-size $GPU_NUM \
        --gpu_memory_utilization $GPU_MEM_B \
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

run_grading() {
    echo ""
    echo "============================================================"
    echo "[STEP 1/1] Grading Rollouts with Model B (Checklist Evaluation)"
    echo "============================================================"
    echo "[INFO] Rollouts: $ROLLOUT_INPUT"
    echo "[INFO] Checklist: $CHECKLIST_DATASET"
    echo "[INFO] Output: $GRADED_OUTPUT"
    echo ""

    local cmd_args=(
        --input "$ROLLOUT_INPUT"
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

    if [[ "$STRIP_THINKING" == "true" ]]; then
        cmd_args+=(--strip_thinking)
    fi
    if [[ -n "$SAMPLE" ]]; then
        cmd_args+=(--sample "$SAMPLE")
    fi

    python "$ROOT/analyze_checklist.py" "${cmd_args[@]}"

    echo "[DONE] Grading complete"
}

# ============================================================================
# Main
# ============================================================================

start_vllm_model_b
wait_for_vllm "$VLLM_LOG_B"
run_grading
stop_vllm

echo ""
echo "============================================================"
echo "[SUCCESS] Grading complete!"
echo "============================================================"
echo "Rollouts: $ROLLOUT_INPUT"
echo "Graded results: $GRADED_OUTPUT"
echo "Metrics: $METRICS_OUTPUT"
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "Partition: $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
