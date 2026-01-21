#!/bin/bash -l


ANNOTATOR_MODEL="Qwen3-30B"
FULL_ANNOTATOR_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
ROOT="/root/code"

VLLM_LOG="$ROOT/vllm_qwen3_30b.log"
CACHE_DIR="$ROOT/vllm_cache"
TIMEOUT=300
DEBUG=0
N_THREADS=128
MAX_INFLIGHT=256
HF_REPO_BASE="JingweiNi/magpie-creative-qwen2.5-augmented"
VLLM_BASE_URL="http://localhost:8000/v1"

export HF_HOME="/iopsstor/scratch/cscs/jni/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_ENCODINGS_BASE="/iopsstor/scratch/cscs/jni/tiktoken_encodings"
huggingface-cli login --token $HUGGINGFACE_TOKEN

VLLM_CMD="vllm serve $FULL_ANNOTATOR_MODEL --tensor-parallel-size 4"

GENERATION_CMD="
INPUT_DATASET_LIST=(
    \"$ROOT/magpie_creative_qwen2.5\"
)

OUTPUT_DATASET_LIST=(
    \"$ROOT/magpie_creative_qwen2.5_augmented\"
)

for i in \${!INPUT_DATASET_LIST[@]}; do
    input_path=\"\${INPUT_DATASET_LIST[i]}\"
    output_path=\"\${OUTPUT_DATASET_LIST[i]}\"
    base_name=\$(basename \"\$output_path\")
    hf_repo=\"${HF_REPO_BASE}-\$base_name\"
    python \"$ROOT/augment_instructions_vllm.py\" \
     --input_dataset \"\$input_path\" \
     --save_to_disk \"\$output_path\" \
     --push_to_hub \"\$hf_repo\" \
     --model \"$FULL_ANNOTATOR_MODEL\" \
     --base_url \"$VLLM_BASE_URL\" \
     --cache_path \"$CACHE_DIR\" \
     --num_workers $N_THREADS \
     --max_inflight $MAX_INFLIGHT
done
"

ENV_CMD="pip install parse nltk sentence-transformers rouge_score  && pip uninstall -y numpy && pip install --no-cache-dir 'numpy==1.26.4'"

if [ $DEBUG -eq 0 ]; then
    LAUNCHING="srun --container-writable --environment=qwen3_next"
else
    LAUNCHING=""
fi

$LAUNCHING bash -lc "
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

  # 3) run env setup + annotation
  echo \"[RUN] Launching annotation...\" 
  
  $GENERATION_CMD
  
  kill -TERM \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
  trap - EXIT
  exit 0
"
