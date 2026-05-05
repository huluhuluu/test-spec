#!/usr/bin/env bash
set -euo pipefail
cd /workspace/code/test-eagle3
export PYTHONUNBUFFERED=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[start] $(date -Iseconds)"
echo "[cwd] $(pwd)"
echo "[python] /root/miniconda3/envs/spec/bin/python"
echo "[gpus] 0 1 2 3"
echo "[speculative] steps=7 topk=10 draft_tokens=32"
echo "[context_length] 1024"
echo "[max_new_tokens] default=512 long=1024"
echo "[backend] AngelSlim=vLLM(official) / Zjcxy+taobao=SGLang(official)"

echo "[phase] download-cmmlu $(date -Iseconds)"
/root/miniconda3/envs/spec/bin/python run_eval.py download-cmmlu

echo "[phase] download-models $(date -Iseconds)"
/root/miniconda3/envs/spec/bin/python run_eval.py download-models

echo "[phase] setup-angelslim-vllm-env $(date -Iseconds)"
bash /workspace/code/test-eagle3/scripts/setup_angelslim_vllm_env.sh eagle3-vllm0112

echo "[phase] prepare-data-or-reuse $(date -Iseconds)"
/root/miniconda3/envs/spec/bin/python run_eval.py prepare-data --sample-size 80 --datasets gsm8k math500 mtbench humaneval ceval cmmlu

echo "[phase] run $(date -Iseconds)"
/root/miniconda3/envs/spec/bin/python run_eval.py run \
  --sample-size 80 \
  --gpus 0 1 2 3 \
  --datasets gsm8k math500 mtbench humaneval ceval cmmlu \
  --models \
    qwen3_1p7b_eagle3 \
    qwen3_4b_eagle3 \
    taobao_qwen3_4b_eagle3 \
    zjcxy_qwen3_4b_eagle3_zh \
    hunyuan_1p8b_eagle3 \
    hunyuan_4b_eagle3

echo "[done] $(date -Iseconds)"
