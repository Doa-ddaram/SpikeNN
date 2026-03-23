"""
Results Analyzer for Merged-Class NCG Experiment
=================================================
Parses the stdout.log from run.py and extracts per-class neuron growth,
difficulty scores, and accuracy to validate that NCG allocates more neurons
to complex (merged) classes.

Usage:
    python analyze_results.py <log_path>
"""

import re
import sys
import numpy as np

CLASS_NAMES = [
    "complex_0 (airplane+auto+truck)",
    "complex_1 (bird+deer+horse)",
    "medium_2  (cat+dog)",
    "simple_3  (frog)",
    "simple_4  (ship)",
]

N_CLASSES = 5


def parse_log(log_path):
    epochs = []
    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Epoch train accuracy
        m = re.match(r"Epoch (\d+): Train Acc ([\d.]+)", line)
        if m:
            epoch = {"epoch": int(m.group(1)), "train_acc": float(m.group(2))}

            # Look ahead for INFO lines
            for j in range(i + 1, min(i + 10, len(lines))):
                info = lines[j].strip()

                # Active neuron info
                m2 = re.search(r"per-class=\[([^\]]+)\]", info)
                if m2:
                    epoch["active_per_class"] = list(map(int, m2.group(1).split(",")))

                # Difficulty and hardness
                m3 = re.search(r"Class difficulty=\[([^\]]+)\]", info)
                if m3:
                    epoch["class_difficulty"] = list(map(float, m3.group(1).replace(",", " ").split()))

                m4 = re.search(r"hardness=\[([^\]]+)\]", info)
                if m4:
                    epoch["class_hardness"] = list(map(float, m4.group(1).replace(",", " ").split()))

                m5 = re.search(r"grew_per_class=\[([^\]]+)\]", info)
                if m5:
                    epoch["grew_per_class"] = list(map(int, m5.group(1).split(",")))

            # Val/test accuracy
            for j in range(i + 1, min(i + 15, len(lines))):
                info = lines[j].strip()
                mv = re.search(r"Accuracy on validation set after epoch \d+: ([\d.]+)", info)
                if mv:
                    epoch["val_acc"] = float(mv.group(1))
                mt = re.search(r"Accuracy on test set after epoch \d+: ([\d.]+)", info)
                if mt:
                    epoch["test_acc"] = float(mt.group(1))

            epochs.append(epoch)

        i += 1

    return epochs


def print_report(epochs):
    if not epochs:
        print("[ERROR] No epoch data found in log.")
        return

    print("=" * 70)
    print("  MERGED-CLASS NCG EXPERIMENT: NEURON ALLOCATION ANALYSIS")
    print("=" * 70)
    print()

    # -- Final epoch summary --
    final = epochs[-1]
    print(f"Total epochs run: {final['epoch'] + 1}")
    if "val_acc" in final:
        print(f"Final val accuracy:  {final['val_acc']:.4f}")
    if "test_acc" in final:
        print(f"Final test accuracy: {final['test_acc']:.4f}")
    print()

    # -- Per-class neuron growth over time --
    print("Active neurons per class over training:")
    print(f"  {'Epoch':>6}  " + "  ".join(f"C{c}({c_name[:8]+'...' if len(c_name)>8 else c_name})" for c, c_name in enumerate(CLASS_NAMES)))
    header = f"  {'Epoch':>6}  " + "  ".join(f"{'C'+str(c):>8}" for c in range(N_CLASSES))
    print(header)
    print("  " + "-" * (8 + 10 * N_CLASSES))
    for ep in epochs:
        if "active_per_class" not in ep:
            continue
        apc = ep["active_per_class"]
        row = f"  {ep['epoch']:>6}  " + "  ".join(f"{v:>8}" for v in apc)
        print(row)
    print()

    # -- Final neuron allocation --
    last_with_active = None
    for ep in reversed(epochs):
        if "active_per_class" in ep:
            last_with_active = ep
            break

    if last_with_active:
        apc = last_with_active["active_per_class"]
        diff = last_with_active.get("class_difficulty", [None] * N_CLASSES)
        print("Final active neurons per class:")
        print(f"  {'Class':<40} {'Active':>8}  {'Difficulty':>12}")
        print("  " + "-" * 64)
        for c in range(N_CLASSES):
            d_str = f"{diff[c]:.3f}" if diff[c] is not None else "N/A"
            marker = " <-- COMPLEX" if c < 2 else (" <-- MEDIUM" if c == 2 else " <-- SIMPLE")
            print(f"  [{c}] {CLASS_NAMES[c]:<36} {apc[c]:>8}  {d_str:>12}{marker}")
        print()
        print(f"  Total active: {sum(apc)} / 200 neurons")
        print()

        # -- Correlation check --
        if all(d is not None for d in diff):
            diff_arr = np.array(diff[:N_CLASSES])
            active_arr = np.array(apc[:N_CLASSES])
            corr = np.corrcoef(diff_arr, active_arr)[0, 1] if len(diff_arr) > 1 else float("nan")
            print(f"  Correlation (difficulty vs active neurons): {corr:.3f}")
            if corr > 0.5:
                print("  => POSITIVE correlation: harder classes get MORE neurons. NCG validated!")
            elif corr > 0.2:
                print("  => WEAK positive correlation. Partial validation.")
            else:
                print("  => Weak or no correlation. Review growth parameters.")
        print()

    # -- Growth events per class --
    total_grew = {c: 0 for c in range(N_CLASSES)}
    for ep in epochs:
        if "grew_per_class" in ep:
            for c, g in enumerate(ep["grew_per_class"]):
                if c < N_CLASSES:
                    total_grew[c] += g

    print("Total neuron growth events per class:")
    for c in range(N_CLASSES):
        marker = " <-- COMPLEX" if c < 2 else (" <-- MEDIUM" if c == 2 else " <-- SIMPLE")
        print(f"  [{c}] {CLASS_NAMES[c]:<36} grew {total_grew[c]} times{marker}")
    print()

    # -- Difficulty trajectory for complex vs simple classes --
    complex_diff = []
    simple_diff = []
    for ep in epochs:
        if "class_difficulty" in ep:
            d = ep["class_difficulty"]
            complex_diff.append(np.mean([d[0], d[1]]))
            simple_diff.append(np.mean([d[3], d[4]]))

    if complex_diff:
        print(f"Avg difficulty (complex classes 0+1): {np.mean(complex_diff):.3f} (mean across epochs)")
        print(f"Avg difficulty (simple classes 3+4):  {np.mean(simple_diff):.3f} (mean across epochs)")
        print(f"Difficulty ratio (complex/simple):    {np.mean(complex_diff)/max(np.mean(simple_diff), 1e-8):.2f}x")

    print()
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_path = "experiments/cifar10-merged-classes/output/stdout.log"
    else:
        log_path = sys.argv[1]

    try:
        epochs = parse_log(log_path)
        print_report(epochs)
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        print("Run the experiment first with: ./run_experiment.sh")
