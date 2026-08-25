#!/bin/bash
# Run an oracle eval session against the sandbox API.
# Usage: ./sandbox-oracle.sh <model> <deg> [extra run_oracle.py flags]
#
# Examples:
#   ./sandbox-oracle.sh qwen3.5:2b alpha-1 --runs 1
#   ./sandbox-oracle.sh qwen3.5:2b alpha-3b --runs 3 --oracle-mode guardian
#   ./sandbox-oracle.sh qwen3:14b alpha-3b --runs 6 --no-think
set -e

MODEL=${1:-qwen3.5:2b}
DEG=${2:-alpha-1}
shift 2 2>/dev/null || true

LABEL="sandbox-${MODEL//[:\/]/-}-${DEG}-$(date +%Y%m%d-%H%M)"

docker exec "${SANDBOX_CONTAINER:-lb-sandbox}" python /app/cli/run_oracle.py \
  --model "$MODEL" \
  --base-url "${LB_BASE_URL:-${BASE_URL:-http://localhost:11434/v1}}" \
  --deg "$DEG" \
  --output "/results/${LABEL}.jsonl" \
  --label "$LABEL" \
  "$@"
