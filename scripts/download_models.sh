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

MODELS=(
  "Qwen/Qwen3-1.7B"
  "AngelSlim/Qwen3-1.7B_eagle3"
  "Qwen/Qwen3-4B"
  "AngelSlim/Qwen3-4B_eagle3"
  "Qwen/Qwen3-4B-Instruct-2507"
  "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3"
  "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh"
  "tencent/Hunyuan-1.8B-Instruct"
  "AngelSlim/Hunyuan-1.8B-Instruct_eagle3"
  "tencent/Hunyuan-4B-Instruct"
  "AngelSlim/Hunyuan-4B-Instruct_eagle3"
)

if [[ ! -f "$HFD" ]]; then
  echo "missing hfd.sh: $HFD" >&2
  echo "edit hfd_path in eval/config.json" >&2
  exit 1
fi

mkdir -p "$MODEL_PATH"
cd "$MODEL_PATH"
for repo in "${MODELS[@]}"; do
  local_dir="$MODEL_PATH/${repo##*/}"
  if [[ -d "$local_dir" ]] && [[ -n "$(find "$local_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[skip-model] $repo -> $local_dir"
    continue
  fi
  echo "[download-model] $repo -> $local_dir"
  bash "$HFD" "$repo" --local-dir "$local_dir"
done
