#!/usr/bin/env bash
export IMAGE_ROOT="/opt/guoxin-asr009"
export PROJECT="$IMAGE_ROOT/project"
export TASK="$PROJECT/tasks/qwen3-asr-bw-vllm"
export PYTHON_BIN="$IMAGE_ROOT/venv/bin/python"
export DATA_DIR="$IMAGE_ROOT/data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export QWEN_ASR_MODEL_PATH="$IMAGE_ROOT/model" QWEN_ASR_PYTHON_BIN="$PYTHON_BIN" QWEN_ASR_PORT=30002
