# Optimized Dynamic Neuron Allocation Experiments (CIFAR-10)

This document defines three practical experiments to improve both efficiency and accuracy
for dynamic neuron allocation in the output layer.

## Common setup

- Input: `ft-extract/extracted/STDP-CSNN/CIFAR10/kfold/0-run1`
- Epochs: 30
- Seeds: 0, 1, 2, 3, 4
- Runner: `app/run.py`

## Experiment A: Growth + Prune

Config: `experiments/cifar10-dynamic/cifar10-200n-dyn-growth-prune.json`

Goal:
- Keep dynamic expansion behavior.
- Periodically remove weakly contributing neurons.

Main knobs to sweep:
- `trainer.dynamic_activation.update_threshold`: 350, 450, 550
- `trainer.pruning.min_target_updates`: 5, 8, 12
- `trainer.pruning.every_n_epochs`: 1, 2
- `trainer.pruning.keep_min_active_per_class`: 5, 6, 7

## Experiment B: Uncertainty-Gated Growth

Config: `experiments/cifar10-dynamic/cifar10-200n-dyn-uncertainty-growth.json`

Goal:
- Grow only on uncertain samples (low winner-vs-second membrane potential margin).
- Reduce unnecessary expansion on easy samples.

Main knobs to sweep:
- `trainer.dynamic_activation.update_threshold`: 400, 500, 600
- `trainer.dynamic_activation.uncertainty_margin`: 0.01, 0.03, 0.05, 0.08

## Experiment C: Budgeted Growth + Prune

Config: `experiments/cifar10-dynamic/cifar10-200n-dyn-budgeted-growth-prune.json`

Goal:
- Enforce strict active-neuron budget per class while keeping recoverable performance.

Main knobs to sweep:
- `trainer.dynamic_activation.max_active_per_class`: 10, 12, 14, 16
- `trainer.dynamic_activation.uncertainty_margin`: 0.02, 0.03, 0.05
- `trainer.pruning.min_target_updates`: 8, 10, 12

## Execution

Run all three experiments over 5 seeds:

```bash
chmod +x experiments/cifar10-dynamic/run_multiseed_optimized_dynamic.sh
./experiments/cifar10-dynamic/run_multiseed_optimized_dynamic.sh
```

## What to compare

For each run, extract from logs:
- Final train/val/test accuracy
- Active neurons per class at each epoch
- Total active neurons at final epoch

Recommended score for model selection:

`score = test_acc - alpha * (final_active_total / 200)`

Use `alpha` in `[0.01, 0.05]` to tune how strongly you penalize neuron usage.
