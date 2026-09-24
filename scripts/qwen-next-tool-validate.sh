#!/usr/bin/env bash
# qwen-next-tool-validate.sh — Execute tool-calling regression battery against live Qwen 3.8 Flash Next
set -eo pipefail

BASE_URL="${1:-http://127.0.0.1:8100/v1}"
TAG="${2:-baseline}"
TRIALS="${3:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "=== Qwen 3.8 Flash Next Tool-Calling Regression Battery ==="
echo "Target URL: $BASE_URL"
echo "Tag:        $TAG"
echo "Trials:     $TRIALS"
echo ""

"$PYTHON" "$SCRIPT_DIR/test_tool_calling_regression.py" \
  --base-url "$BASE_URL" \
  --model-name "qwen3.8-flash-next-nvfp4" \
  --tag "$TAG" \
  --trials "$TRIALS" \
  --temperature 0.0 \
  --max-tokens 1024

echo ""
echo "=== Tool-Calling Regression Battery Completed ==="
