#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY_ROOT/env.sh"
exec "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/submission009/benchmark_asr.py" --data "$DATA_DIR" --output "${BENCH_OUTPUT:-$REPOSITORY_ROOT/bench-replay-$(date -u +%Y%m%dT%H%M%SZ)}" "$@"
