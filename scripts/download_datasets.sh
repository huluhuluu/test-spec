#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

readarray -t CONFIG_LINES < <(python - <<'PY'
from eval.config_loader import DATA_PATH, HF_DATASET_REPOS, HF_ENDPOINT, HFD_PATH, repo_leaf
print(DATA_PATH.expanduser())
print(HFD_PATH.expanduser())
print(HF_ENDPOINT)
for name, repo in HF_DATASET_REPOS.items():
    print(f"{name}\t{repo}\t{DATA_PATH.expanduser() / repo_leaf(repo)}")
PY
)

DATA_PATH="${CONFIG_LINES[0]}"
HFD="${CONFIG_LINES[1]}"
HF_ENDPOINT="${CONFIG_LINES[2]}"
export HF_ENDPOINT
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

if [[ ! -f "$HFD" ]]; then
  echo "missing hfd.sh: $HFD" >&2
  echo "edit downloads.hfd_path in eval/config.json" >&2
  exit 1
fi

mkdir -p "$DATA_PATH"

for line in "${CONFIG_LINES[@]:3}"; do
  IFS=$'\t' read -r name repo local_dir <<< "$line"
  if [[ -d "$local_dir" ]] && [[ -n "$(find "$local_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[skip-dataset] $name -> $local_dir"
    continue
  fi
  echo "[download-dataset] $name -> $local_dir"
  bash "$HFD" "$repo" --dataset --local-dir "$local_dir"
done

if [[ -d "third_party/CMMLU/data/test" ]] && [[ -n "$(find third_party/CMMLU/data/test -name '*.csv' -print -quit)" ]]; then
  echo "[skip-dataset] cmmlu -> third_party/CMMLU/data/test"
else
  echo "[download-dataset] cmmlu -> third_party/CMMLU"
  git submodule update --init --recursive --depth 1 -- third_party/CMMLU
fi
