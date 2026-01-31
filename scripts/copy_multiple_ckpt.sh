#!/usr/bin/env bash
set -euo pipefail

# Default model names (FIXED: removed erroneous inner braces)
MODEL_A_NAME="${1:-Qwen3-30B}"
MODEL_B_NAME="${2:-Qwen3-235B}"

# Validate env var exists
if [[ -z "${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR:-}" ]]; then
    echo "ERROR: PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR is not set" >&2
    exit 1
fi

# Helper function to extract path safely (handles regex chars)
get_model_path() {
    local model_name="$1"
    local env_value="$PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR"
    
    # Escape regex special chars (., -, etc.) for literal matching
    local escaped_name
    escaped_name=$(printf '%s' "$model_name" | sed 's/[][\\.^$*+?(){}|]/\\&/g')
    
    # Pass safely to awk via -v (FIXED: proper variable interpolation)
    echo "$env_value" | awk -F';' -v name="$escaped_name" '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ ("^" name ":")) {
                    sub(("^" name ":"), "", $i)
                    print $i
                    exit
                }
            }
        }'
}

# Extract paths (FIXED: using helper function)
MODEL_A_PATH=$(get_model_path "$MODEL_A_NAME")
MODEL_B_PATH=$(get_model_path "$MODEL_B_NAME")

# Validate paths found
if [[ -z "$MODEL_A_PATH" ]]; then
    echo "ERROR: Model '$MODEL_A_NAME' not found in PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ -z "$MODEL_B_PATH" ]]; then
    echo "ERROR: Model '$MODEL_B_NAME' not found in PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR" >&2
    exit 1
fi

echo "Cloning $MODEL_A_NAME from $MODEL_A_PATH"
rclone copy --progress --transfers=50 "$MODEL_A_PATH" "/root/models/${MODEL_A_NAME}"

echo "Cloning $MODEL_B_NAME from $MODEL_B_PATH"
rclone copy --progress --transfers=50 "$MODEL_B_PATH" "/root/models/${MODEL_B_NAME}"

echo "✓ Done!"