#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/workspace/code/test-eagle3/artifacts/external/CMMLU}"

export http_proxy=http://192.168.124.101:7890
export https_proxy=http://192.168.124.101:7890
git clone --depth 1 https://github.com/haonan-li/CMMLU.git "$TARGET_DIR"
unset http_proxy
unset https_proxy
