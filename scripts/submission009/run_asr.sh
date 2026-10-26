#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_ROOT="${ASR009_IMAGE_ROOT:-/opt/guoxin-asr009}"
TASK="$IMAGE_ROOT/project/tasks/qwen3-asr-bw-vllm"
export QWEN_ASR_MODEL_PATH="$IMAGE_ROOT/model"
export QWEN_ASR_PYTHON_BIN="$IMAGE_ROOT/venv/bin/python"
export QWEN_ASR_MAX_RUNTIME_SECONDS="${ASR009_MAX_RUNTIME_SECONDS:-9900}"
export QWEN_ASR_LOAD_TIMEOUT=900
export QWEN_ASR_PORT=30002
source "$TASK/env.sh"
export PYTHONPATH="$IMAGE_ROOT/project/scripts:$TASK/compat${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=hcu,qwen_asr_tf55_compat
exec "$QWEN_ASR_PYTHON_BIN" "$IMAGE_ROOT/project/scripts/recon2reason_runtime.py" launch --task-dir "$TASK" "$@"
