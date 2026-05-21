#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

readarray -t CONFIG_PATHS < <(python - <<'PY'
from eval.config_loader import HF_ENDPOINT, HFD_PATH, MODEL_PATH
print(MODEL_PATH.expanduser())
print(HFD_PATH.expanduser())
print(HF_ENDPOINT)
PY
)
MODEL_PATH="${CONFIG_PATHS[0]}"
HFD="${CONFIG_PATHS[1]}"
HF_ENDPOINT="${CONFIG_PATHS[2]}"
export HF_ENDPOINT
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

MODELS=(
  # "Qwen/Qwen3-1.7B"
  # "AngelSlim/Qwen3-1.7B_eagle3"
  # "Qwen/Qwen3-4B"
  # "AngelSlim/Qwen3-4B_eagle3"
  # "Qwen/Qwen3-4B-Instruct-2507"
  # "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3"
  # "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh"
  "tencent/Hunyuan-1.8B-Instruct"
  "AngelSlim/Hunyuan-1.8B-Instruct_eagle3"
  "taobao-mnn/Hunyuan-1.8B-Instruct-MNN"
  # "tencent/Hunyuan-4B-Instruct"
  # "AngelSlim/Hunyuan-4B-Instruct_eagle3"
)

if [[ ! -f "$HFD" ]]; then
  echo "missing hfd.sh: $HFD" >&2
  echo "edit downloads.hfd_path in eval/config.json" >&2
  exit 1
fi

mkdir -p "$MODEL_PATH"
cd "$MODEL_PATH"
for repo in "${MODELS[@]}"; do
  local_dir="$MODEL_PATH/${repo##*/}"
  if [[ -d "$local_dir" ]]; then
    if [[ -n "$(find "$local_dir" -name '*.aria2' -print -quit)" ]]; then
      echo "[resume-model] $repo -> $local_dir"
    elif [[ -f "$local_dir/config.json" ]] && find "$local_dir" -name '*.safetensors' -print -quit | grep -q .; then
      echo "[skip-model] $repo -> $local_dir"
      continue
    else
      echo "[repair-model] $repo -> $local_dir"
    fi
  else
    echo "[download-model] $repo -> $local_dir"
  fi
  bash "$HFD" "$repo" --local-dir "$local_dir"
done
