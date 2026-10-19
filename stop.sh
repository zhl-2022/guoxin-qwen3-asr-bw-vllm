#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPOSITORY_ROOT/tasks/qwen3-asr-bw-vllm/stop.sh" "$@"
