#!/bin/bash -l


echo $PRIMUS_OUTPUT_DIR
OSS_SAVE_PATH="/primus_datasets/jingwei"

FULL_ANNOTATOR_MODEL="./models/Qwen3-235B-A22B-Thinking-2507-FP8"
ROOT="/root/code"

VLLM_LOG="$ROOT/vllm_qwen3_235b.log"
TIMEOUT=600
DEBUG=0
N_THREADS=64
MAX_INFLIGHT=128
REPO_NAME="magpie_creative_dedup_checklist"
CACHE_DIR="$ROOT/vllm_cache_qwen3_235b_$REPO_NAME"
OUTPUT_DIR="$PRIMUS_OUTPUT_DIR/$REPO_NAME"
VLLM_BASE_URL="http://localhost:8000/v1"
GPU_NUM=8
SYSTEM_PROMPT_PATH="$ROOT/prompt/checklist_extraction.txt"
USER_PROMPT_PATH="$ROOT/prompt/checklist_extraction_user.txt"
SPLIT="train"
_RAW_GENERATION_CONFIG='{"temperature": 0.6, "top_p": 0.95, "extra_body": {"enable_thinking": true, "top_k": 20}}'
GENERATION_CONFIG_ESCAPED=$(printf '%q' "$_RAW_GENERATION_CONFIG")
GPU_MEM_UTILIZATION=0.7


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
huggingface-cli login --token $HUGGINGFACE_TOKEN

VLLM_CMD="vllm serve $FULL_ANNOTATOR_MODEL --tensor-parallel-size $GPU_NUM --enable-expert-parallel --gpu_memory_utilization $GPU_MEM_UTILIZATION"

GENERATION_CMD="
INPUT_DATASET_LIST=(
    \"JingweiNi/magpie_creative_dedup\"
)

OUTPUT_DATASET_LIST=(
    \"$OUTPUT_DIR\"
)

for i in \${!INPUT_DATASET_LIST[@]}; do
    python \"$ROOT/checklist_extraction_vllm.py\" \
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
     --generation_config $GENERATION_CONFIG_ESCAPED

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

  # 3) run env setup + checklist extraction
  echo \"[RUN] Launching checklist extraction...\"
  
  $GENERATION_CMD

  cp -r $CACHE_DIR $PRIMUS_OUTPUT_DIR
  cp -r $OUTPUT_DIR $OSS_SAVE_PATH
  cp -r $CACHE_DIR $PRIMUS_OUTPUT_DIR
  
  kill -TERM \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
  trap - EXIT
  exit 0
"
