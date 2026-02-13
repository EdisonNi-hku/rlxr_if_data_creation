#!/bin/bash -l
#
# Quality Analysis Pipeline
#
# Runs LLM-based quality analysis on two model rollouts, then merges the
# quality scores back into enriched JSONL files (one per model).
#
# The pipeline:
#   1. (Optional) Starts a vLLM server
#   2. Runs analyze_quality_vllm.py  — pointwise grading of each response
#   3. Merges quality columns into the graded JSONL files
#
# Output per model:  <rollout_basename>_quality.jsonl
#   = original graded columns + quality dimension scores + quality_notes
#

set -euo pipefail

# ============================================================================
# >>> CONFIGURE THESE <<<
# ============================================================================

# --- Input files (REQUIRED — set before running) ---
M1_ROLLOUT=${M1_ROLLOUT:-"checkpoint_eval/eval_Qwen3-30B-GDPO-150.jsonl"}
M1_GRADED=${M1_GRADED:-"checkpoint_eval/eval_Qwen3-30B-GDPO-150_graded.jsonl"}
M2_ROLLOUT=${M2_ROLLOUT:-"checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150.jsonl"}
M2_GRADED=${M2_GRADED:-"checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150_graded.jsonl"}

# --- LLM model configuration ---
MODEL=${MODEL:-"openai/gpt-oss-120b"}
GENERATION_CONFIG=${GENERATION_CONFIG:-'{"extra_body": {"reasoning_effort": "medium"}}'}

# --- vLLM server ---
#   Set START_VLLM=true to have this script launch & manage a local server.
#   Set START_VLLM=false if you already have a server running.
START_VLLM=${START_VLLM:-false}
VLLM_BASE_URL=${VLLM_BASE_URL:-"http://localhost:8000/v1"}
GPU_NUM=${GPU_NUM:-8}
GPU_MEM=${GPU_MEM:-0.9}
VLLM_LOG=${VLLM_LOG:-"vllm_quality.log"}
VLLM_TIMEOUT=${VLLM_TIMEOUT:-600}

# --- Processing ---
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_INFLIGHT=${MAX_INFLIGHT:-32}
MAX_SAMPLES=${MAX_SAMPLES:-""}          # empty = all
START_INDEX=${START_INDEX:-0}
NO_SYSTEM=${NO_SYSTEM:-false}

# --- Partitioning (for distributed runs) ---
PARTITION_NUM=${PARTITION_NUM:-${NNODES:-1}}
PARTITION_INDEX=${PARTITION_INDEX:-${RANK:-0}}

# --- Output ---
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dirname "$M1_ROLLOUT")"}
ANALYSIS_OUTPUT=${ANALYSIS_OUTPUT:-"$OUTPUT_DIR/quality_analysis.jsonl"}
CACHE_PATH=${CACHE_PATH:-"$HOME/.cache"}

# ============================================================================
# Derived paths (you usually don't need to touch these)
# ============================================================================

M1_BASENAME="$(basename "${M1_GRADED%.jsonl}")"
M2_BASENAME="$(basename "${M2_GRADED%.jsonl}")"
M1_QUALITY_OUTPUT="$OUTPUT_DIR/${M1_BASENAME}_quality.jsonl"
M2_QUALITY_OUTPUT="$OUTPUT_DIR/${M2_BASENAME}_quality.jsonl"

# ============================================================================
# Print configuration
# ============================================================================

echo "[INFO] Quality Analysis Pipeline"
echo "  M1 rollout:   $M1_ROLLOUT"
echo "  M1 graded:    $M1_GRADED"
echo "  M2 rollout:   $M2_ROLLOUT"
echo "  M2 graded:    $M2_GRADED"
echo "  Model:        $MODEL"
echo "  vLLM URL:     $VLLM_BASE_URL"
echo "  Start vLLM:   $START_VLLM"
echo "  Workers:      $NUM_WORKERS"
echo "  Max inflight: $MAX_INFLIGHT"
echo "  Partition:    $PARTITION_INDEX of $PARTITION_NUM"
echo "  Analysis out: $ANALYSIS_OUTPUT"
echo "  M1 quality:   $M1_QUALITY_OUTPUT"
echo "  M2 quality:   $M2_QUALITY_OUTPUT"
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "  Max samples:  $MAX_SAMPLES"
fi
echo ""

# ============================================================================
# vLLM server management
# ============================================================================

VLLM_PID=""

start_vllm() {
    echo "[INFO] Starting vLLM server..."
    vllm serve "$MODEL" \
        --tensor-parallel-size "$GPU_NUM" \
        --gpu_memory_utilization "$GPU_MEM" \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "[INFO] vLLM PID=$VLLM_PID"
}

wait_for_vllm() {
    echo "[WAIT] Watching $VLLM_LOG for readiness..."
    if ! timeout "$VLLM_TIMEOUT" bash -c "( tail -n0 -f \"$VLLM_LOG\" & ) | grep -q -- 'Application startup complete.'"; then
        echo "[ERROR] vLLM did not become ready within ${VLLM_TIMEOUT}s"
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

# ============================================================================
# Step 1: Run quality analysis
# ============================================================================

run_analysis() {
    echo ""
    echo "============================================================"
    echo "[STEP 1/2] Running LLM quality analysis"
    echo "============================================================"

    local cmd_args=(
        --model1-rollout "$M1_ROLLOUT"
        --model1-graded  "$M1_GRADED"
        --model2-rollout "$M2_ROLLOUT"
        --model2-graded  "$M2_GRADED"
        --output         "$ANALYSIS_OUTPUT"
        --model          "$MODEL"
        --base_url       "$VLLM_BASE_URL"
        --cache_path     "$CACHE_PATH"
        --num_workers    "$NUM_WORKERS"
        --max_inflight   "$MAX_INFLIGHT"
        --start_index    "$START_INDEX"
        --generation_config "$GENERATION_CONFIG"
        --partition_num  "$PARTITION_NUM"
        --partition_index "$PARTITION_INDEX"
    )

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd_args+=(--max_samples "$MAX_SAMPLES")
    fi
    if [[ "$NO_SYSTEM" == "true" ]]; then
        cmd_args+=(--no_system)
    fi

    python "$ROOT/analyze_quality_vllm.py" "${cmd_args[@]}"

    echo "[DONE] Analysis complete: $ANALYSIS_OUTPUT"
}

# ============================================================================
# Step 2: Merge quality columns into graded JSONL
# ============================================================================

merge_quality() {
    echo ""
    echo "============================================================"
    echo "[STEP 2/2] Merging quality scores into graded files"
    echo "============================================================"

    python -c "
import json, sys

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

DIMS = [
    'incoherent_expression',
    'logical_inconsistency',
    'inappropriate_word_choice',
    'repetitive_expression',
    'language_inconsistency',
]

analysis = load_jsonl('$ANALYSIS_OUTPUT')
analysis_by_idx = {row['idx']: row for row in analysis}

m1_graded = load_jsonl('$M1_GRADED')
m2_graded = load_jsonl('$M2_GRADED')

def enrich(graded_rows, analysis_key, output_path):
    with open(output_path, 'w') as f:
        for i, row in enumerate(graded_rows):
            a = analysis_by_idx.get(i, {}).get(analysis_key, {})
            for dim in DIMS:
                row['quality_' + dim] = a.get(dim, -1)
            row['quality_notes'] = a.get('notes', '')
            row['quality_any_issue'] = int(any(a.get(d, 0) == 1 for d in DIMS))
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    n_issues = sum(1 for i in range(len(graded_rows))
                   if analysis_by_idx.get(i, {}).get(analysis_key, {})
                   and any(analysis_by_idx[i][analysis_key].get(d, 0) == 1 for d in DIMS))
    print(f'  {output_path}: {len(graded_rows)} rows, {n_issues} with quality issues')

enrich(m1_graded, 'm1_analysis', '$M1_QUALITY_OUTPUT')
enrich(m2_graded, 'm2_analysis', '$M2_QUALITY_OUTPUT')
"

    echo "[DONE] Enriched files written"
}

# ============================================================================
# Main
# ============================================================================

if [[ "$START_VLLM" == "true" ]]; then
    trap 'stop_vllm' EXIT
    start_vllm
    wait_for_vllm
fi

mkdir -p "$OUTPUT_DIR"

run_analysis
merge_quality

if [[ "$START_VLLM" == "true" ]]; then
    stop_vllm
    trap - EXIT
fi

echo ""
echo "============================================================"
echo "[SUCCESS] Quality analysis pipeline complete!"
echo "============================================================"
echo "  Analysis:   $ANALYSIS_OUTPUT"
echo "  M1 quality: $M1_QUALITY_OUTPUT"
echo "  M2 quality: $M2_QUALITY_OUTPUT"
if [[ "$PARTITION_NUM" -gt 1 ]]; then
    echo "  Partition:  $PARTITION_INDEX of $PARTITION_NUM"
fi
echo ""

exit 0
