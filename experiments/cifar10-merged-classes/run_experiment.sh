#!/usr/bin/env zsh
# =============================================================================
# Experiment: Uncertainty-based NCG validation with merged CIFAR10 classes
#
# Design:
#   Class 0 (COMPLEX, 3 merged): airplane + automobile + truck
#   Class 1 (COMPLEX, 3 merged): bird + deer + horse
#   Class 2 (MEDIUM,  2 merged): cat + dog
#   Class 3 (SIMPLE,  1 class):  frog
#   Class 4 (SIMPLE,  1 class):  ship
#
# Hypothesis: NCG allocates more active neurons to complex/hard classes (0, 1)
#             due to higher intra-class variance -> higher classification difficulty.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
IN="$SCRIPT_DIR/input"
OUT="$SCRIPT_DIR/output"

cd "$REPO_ROOT"

# Step 1: Prepare merged dataset
if [ ! -f "$IN/trainset.npy" ]; then
    echo "=== [Step 1] Preparing merged CIFAR10 dataset ==="
    WANDB_MODE=disabled "$PY" "$SCRIPT_DIR/prepare_dataset.py"
else
    echo "=== [Step 1] Dataset already exists, skipping preparation ==="
fi

# Step 2: Run NCG experiment
echo ""
echo "=== [Step 2] Running NCG experiment with merged classes ==="
rm -rf "$OUT"
mkdir -p "$OUT"

WANDB_MODE=disabled "$PY" app/run.py \
    "$IN" "$OUT" \
    "$SCRIPT_DIR/config-ncg.json" \
    --seed 0 \
    --run_name "cifar10-merged-ncg-seed0" \
    2>&1 | tee "$OUT/stdout.log"

# Step 3: Analyze results
echo ""
echo "=== [Step 3] Analyzing results ==="
WANDB_MODE=disabled "$PY" "$SCRIPT_DIR/analyze_results.py" "$OUT/stdout.log"

echo ""
echo "=== Experiment complete. Results in: $OUT ==="
