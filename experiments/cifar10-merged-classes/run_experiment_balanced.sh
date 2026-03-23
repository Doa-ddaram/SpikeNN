#!/usr/bin/env zsh
# Balanced version: all classes have equal sample count (4500 each)
# so intra-class variance alone drives difficulty differences.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
IN="$SCRIPT_DIR/input-balanced"
OUT="$SCRIPT_DIR/output-balanced"

cd "$REPO_ROOT"

if [ ! -f "$IN/trainset.npy" ]; then
    echo "=== [Step 1] Preparing balanced merged CIFAR10 dataset ==="
    WANDB_MODE=disabled "$PY" "$SCRIPT_DIR/prepare_dataset_balanced.py"
else
    echo "=== [Step 1] Balanced dataset already exists, skipping ==="
fi

echo ""
echo "=== [Step 2] Running NCG experiment (balanced classes) ==="
rm -rf "$OUT"
mkdir -p "$OUT"

WANDB_MODE=disabled "$PY" app/run.py \
    "$IN" "$OUT" \
    "$SCRIPT_DIR/config-ncg.json" \
    --seed 0 \
    --run_name "cifar10-merged-ncg-balanced-seed0" \
    2>&1 | tee "$OUT/stdout.log"

echo ""
echo "=== [Step 3] Analyzing results ==="
WANDB_MODE=disabled "$PY" "$SCRIPT_DIR/analyze_results.py" "$OUT/stdout.log"

echo ""
echo "=== Done. Results in: $OUT ==="
