#!/usr/bin/env zsh
set -euo pipefail

BASE="logs/STDP-CSNN/CIFAR10/multiseed-optimized-dynamic-v2"
IN="ft-extract/extracted/STDP-CSNN/CIFAR10/kfold/0-run1"
PY="/Users/sangkiko/SpikeNN/.venv/bin/python"

mkdir -p "$BASE"

# v2 tuning notes:
# 1) growth trigger was too conservative in v1, so update_threshold was reduced.
# 2) growth is delayed by 2 epochs (growth_after_epoch) to avoid noisy early updates.
# 3) uncertainty margin is widened so uncertainty-gated growth can actually fire.
cases=(
  "growth-prune-v2 experiments/cifar10-dynamic/cifar10-200n-dyn-growth-prune-v2.json"
  "uncertainty-growth-v2 experiments/cifar10-dynamic/cifar10-200n-dyn-uncertainty-growth-v2.json"
  "budgeted-growth-prune-v2 experiments/cifar10-dynamic/cifar10-200n-dyn-budgeted-growth-prune-v2.json"
)

for s in 0; do
  for case in "${cases[@]}"; do
    name="${case%% *}"
    cfg="${case#* }"
    out="$BASE/${name}-seed${s}"

    echo "=== seed ${s}: ${name} ==="
    rm -rf "$out"
    mkdir -p "$out"

    WANDB_MODE=disabled "$PY" app/run.py "$IN" "$out" "$cfg" \
      --seed "$s" \
      --run_name "${name}-seed${s}" \
      2>&1 | tee "$out/stdout.log"
  done
done

echo "all-runs-complete"
