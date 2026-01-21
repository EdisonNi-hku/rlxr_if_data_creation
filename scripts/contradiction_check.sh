#!/bin/bash -l

ANNOTATOR_MODEL="Qwen3-32B"
FULL_ANNOTATOR_MODEL="./models/Qwen3-32B"
ROOT="/root/code"

VLLM_LOG="$ROOT/vllm_qwen3_32b.log"
CACHE_DIR="$ROOT/vllm_cache"
TIMEOUT=600
DEBUG=0
N_THREADS=128
MAX_INFLIGHT=256
REPO_NAME="magpie_creative_dedup_verifiable_if_filtered"
VLLM_BASE_URL="http://localhost:8000/v1"
GPU_NUM=8
SYSTEM_PROMPT_PATH="$ROOT/prompt/contradiction_check.txt"
USER_PROMPT_PATH="$ROOT/prompt/constradiction_check_user.txt"
SPLIT="train"
CACHE_PATH="$ROOT/.cache"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
huggingface-cli login --token $HUGGINGFACE_TOKEN

VLLM_CMD="vllm serve $FULL_ANNOTATOR_MODEL --tensor-parallel-size $GPU_NUM"

GENERATION_CMD="
INPUT_DATASET_LIST=(
    \"JingweiNi/magpie_creative_dedup_verifiable_if\"
)

OUTPUT_DATASET_LIST=(
    \"$ROOT/$REPO_NAME\"
)

for i in \${!INPUT_DATASET_LIST[@]}; do
    python \"$ROOT/contradiction_check_vllm.py\" \
     --input_dataset \"\${INPUT_DATASET_LIST[i]}\" \
     --save_to_disk \"\${OUTPUT_DATASET_LIST[i]}\" \
     --push_to_hub \"JingweiNi/$REPO_NAME\" \
     --model \"$FULL_ANNOTATOR_MODEL\" \
     --base_url \"$VLLM_BASE_URL\" \
     --cache_path \"$CACHE_DIR\" \
     --num_workers $N_THREADS \
     --max_inflight $MAX_INFLIGHT \
     --system_prompt_path $SYSTEM_PROMPT_PATH \
     --user_prompt_path $USER_PROMPT_PATH \
     --split $SPLIT \
     --cache_path $CACHE_PATH

done
"

bash -lc "
  set -euo pipefail

  $ENV_CMD

  # 1) start vLLM in background
  $VLLM_CMD > \"$VLLM_LOG\" 2>&1 &
  VLLM_PID=\$!
  echo \"[INFO] vLLM PID=\$VLLM_PID\"

  # Ensure cleanup on any exit
  trap 'echo \"[CLEANUP] Stopping vLLM (\$VLLM_PID)\"; kill -TERM \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true' EXIT

  # 2) wait for readiness (timeout ${TIMEOUT}s)
  echo \"[WAIT] Watching $VLLM_LOG for readiness...\"
  if ! timeout $TIMEOUT bash -c '( tail -n0 -f \"$VLLM_LOG\" & ) | grep -q -- \"Application startup complete.\"'; then
    echo \"[ERROR] vLLM did not become ready within ${TIMEOUT}s\"
    exit 1
  fi
  echo \"[READY] vLLM is ready.\"

  # 3) run env setup + contradiction check
  echo \"[RUN] Launching contradiction check...\"
  
  $GENERATION_CMD
  
  kill -TERM \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
  trap - EXIT
  exit 0
"
