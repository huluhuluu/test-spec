#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-eagle3-vllm0112}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.11.2}"

source /root/miniconda3/etc/profile.d/conda.sh
CONDA_ROOT="/root/miniconda3"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[reuse-env] $ENV_NAME"
else
  echo "[create-env] $ENV_NAME python=$PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

echo "[install] vllm==$VLLM_VERSION"
"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
PIP_NO_CACHE_DIR=1 "$ENV_PREFIX/bin/python" -m pip install --no-cache-dir "vllm==$VLLM_VERSION"

echo "[verify]"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
import vllm
print("python_ok")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY
