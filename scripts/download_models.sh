#!/usr/bin/env bash
set -euo pipefail

HF_HOME=/data/HUGGINGFACE
HFD="$HF_HOME/hfd.sh"
HF_ENDPOINT=https://hf-mirror.com
export HF_HOME HF_ENDPOINT

MODELS=(
  "Qwen/Qwen3-1.7B"
  "AngelSlim/Qwen3-1.7B_eagle3"
  "Qwen/Qwen3-4B-Instruct-2507"
  "AngelSlim/Qwen3-4B_eagle3"
  "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3"
  "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh"
  "tencent/Hunyuan-1.8B-Instruct"
  "AngelSlim/Hunyuan-1.8B-Instruct_eagle3"
  "tencent/Hunyuan-4B-Instruct"
  "AngelSlim/Hunyuan-4B-Instruct_eagle3"
)

cd "$HF_HOME"
for repo in "${MODELS[@]}"; do
  local_dir="$HF_HOME/${repo##*/}"
  bash "$HFD" "$repo" --local-dir "$local_dir"
done
