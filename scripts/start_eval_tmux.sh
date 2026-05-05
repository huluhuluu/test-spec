#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/code/test-eagle3"
PYTHON_BIN="/root/miniconda3/envs/spec/bin/python"
SESSION_NAME="${1:-eagle3_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/artifacts/tmux_logs"
RUN_LOG="$LOG_DIR/${SESSION_NAME}.log"
PID_LOG="$LOG_DIR/${SESSION_NAME}.pid"
RUNNER_SCRIPT="$LOG_DIR/${SESSION_NAME}.runner.sh"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

cat > "$RUNNER_SCRIPT" <<'EOF'
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
EOF

chmod +x "$RUNNER_SCRIPT"
tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "bash '$RUNNER_SCRIPT' 2>&1 | tee -a '$RUN_LOG'" C-m
tmux list-panes -t "$SESSION_NAME" -F "#{pane_pid}" > "$PID_LOG"

echo "SESSION_NAME=$SESSION_NAME"
echo "RUN_LOG=$RUN_LOG"
echo "PID_LOG=$PID_LOG"
echo "RUNNER_SCRIPT=$RUNNER_SCRIPT"
