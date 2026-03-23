"""
NCG vs Static Neuron Allocation (SoftHebb features): Comparison Visualization
==============================================================================
Usage:
    python compare_experiments_softhebb.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES_SHORT = [
    "complex_0\n(air+auto+truck)",
    "complex_1\n(bird+deer+horse)",
    "medium_2\n(cat+dog)",
    "simple_3\n(frog)",
    "simple_4\n(ship)",
]
N_CLASSES = 5

RUNS = [
    (
        "NCG-Selective (dynamic)",
        os.path.join(SCRIPT_DIR, "output-softhebb-ncg", "stdout.log"),
        "#2196F3",
        "-",
    ),
    (
        "Static-100 (20/class fixed)",
        os.path.join(SCRIPT_DIR, "output-softhebb-static100", "stdout.log"),
        "#9C27B0",
        ":",
    ),
    (
        "Static-60 (12/class fixed)",
        os.path.join(SCRIPT_DIR, "output-softhebb-static60", "stdout.log"),
        "#FF5722",
        "--",
    ),
    (
        "Static-200 (40/class fixed)",
        os.path.join(SCRIPT_DIR, "output-softhebb-static200", "stdout.log"),
        "#4CAF50",
        "-.",
    ),
]

STATIC_NEURONS = {
    "Static-100 (20/class fixed)": 100,
    "Static-60 (12/class fixed)": 60,
    "Static-200 (40/class fixed)": 200,
}


def parse_log(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        lines = f.readlines()

    epoch_map = {}
    for line in lines:
        line = line.strip()

        m = re.match(r"Epoch (\d+): Train Acc ([\d.]+)", line)
        if m:
            ep = int(m.group(1))
            epoch_map.setdefault(ep, {})["epoch"] = ep
            epoch_map[ep]["train_acc"] = float(m.group(2))

        m2 = re.search(r"per-class=\[([^\]]+)\]", line)
        if m2 and epoch_map:
            ep = max(epoch_map)
            epoch_map[ep]["active_per_class"] = list(map(int, m2.group(1).split(",")))

        m3 = re.search(r"Class difficulty=\[([^\]]+)\]", line)
        if m3 and epoch_map:
            ep = max(epoch_map)
            epoch_map[ep]["class_difficulty"] = list(
                map(float, m3.group(1).replace(",", " ").split())
            )

        mv = re.search(r"Accuracy on validation set after epoch (\d+): ([\d.]+)", line)
        if mv:
            ep = int(mv.group(1))
            epoch_map.setdefault(ep, {})["val_acc"] = float(mv.group(2))

        mt = re.search(r"Accuracy on test set after epoch (\d+): ([\d.]+)", line)
        if mt:
            ep = int(mt.group(1))
            epoch_map.setdefault(ep, {})["test_acc"] = float(mt.group(2))

    return [epoch_map[k] for k in sorted(epoch_map)]


def get_final_active(epochs, n_neurons_total=None):
    for ep in reversed(epochs):
        if "active_per_class" in ep:
            return ep["active_per_class"]
    if n_neurons_total is not None:
        per = n_neurons_total // N_CLASSES
        return [per] * N_CLASSES
    return None


def main():
    all_data = {}
    for label, log_path, color, ls in RUNS:
        epochs = parse_log(log_path)
        all_data[label] = {"epochs": epochs, "color": color, "ls": ls}
        print(f"[{label}] epochs parsed: {len(epochs)}  (log: {log_path})")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "NCG vs Static Neuron Allocation — Merged CIFAR10 (SoftHebb Features)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_val   = fig.add_subplot(gs[0, 1])
    ax_test  = fig.add_subplot(gs[0, 2])
    ax_active = fig.add_subplot(gs[1, 0:2])
    ax_diff   = fig.add_subplot(gs[1, 2])

    ax_train.set_title("Train Accuracy")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Accuracy")

    ax_val.set_title("Validation Accuracy")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Accuracy")

    ax_test.set_title("Test Accuracy")
    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("Accuracy")

    for label, data in all_data.items():
        epochs = data["epochs"]
        color = data["color"]
        ls = data["ls"]
        if not epochs:
            print(f"  [WARNING] no data for {label}, skipping plot")
            continue

        xs = [e["epoch"] for e in epochs]
        train_acc = [e.get("train_acc", float("nan")) for e in epochs]
        val_acc   = [e.get("val_acc",   float("nan")) for e in epochs]
        test_acc  = [e.get("test_acc",  float("nan")) for e in epochs]

        ax_train.plot(xs, train_acc, color=color, linestyle=ls, linewidth=2, label=label, marker="o", markersize=3)
        ax_val.plot(  xs, val_acc,   color=color, linestyle=ls, linewidth=2, label=label, marker="o", markersize=3)
        ax_test.plot( xs, test_acc,  color=color, linestyle=ls, linewidth=2, label=label, marker="o", markersize=3)

    for ax in [ax_train, ax_val, ax_test]:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    all_accs = []
    for data in all_data.values():
        for e in data["epochs"]:
            for key in ("train_acc", "val_acc", "test_acc"):
                v = e.get(key, float("nan"))
                if not np.isnan(v):
                    all_accs.append(v)
    if all_accs:
        y_min = max(0.0, min(all_accs) - 0.03)
        y_max = min(1.0, max(all_accs) + 0.03)
        for ax in [ax_train, ax_val, ax_test]:
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

    # ---- Active neurons per class ----
    ax_active.set_title("Final Active Neurons per Class")
    ax_active.set_ylabel("# Active Neurons")
    x = np.arange(N_CLASSES)
    n_runs = len(all_data)
    bar_width = 0.8 / n_runs
    offsets = [bar_width * (i - (n_runs - 1) / 2) for i in range(n_runs)]

    for idx, (label, data) in enumerate(all_data.items()):
        n_total = STATIC_NEURONS.get(label)
        apc = get_final_active(data["epochs"], n_total)
        if apc is None:
            continue
        bars = ax_active.bar(
            x + offsets[idx], apc[:N_CLASSES], bar_width * 0.9,
            label=label, color=data["color"], alpha=0.8, edgecolor="white",
        )
        for bar, val in zip(bars, apc[:N_CLASSES]):
            ax_active.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=7,
            )

    ax_active.set_xticks(x)
    ax_active.set_xticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax_active.legend(fontsize=8)
    ax_active.grid(True, axis="y", alpha=0.3)
    ax_active.axvspan(-0.5, 1.5, alpha=0.05, color="red")
    ax_active.axvspan(1.5,  2.5, alpha=0.05, color="yellow")
    ax_active.axvspan(2.5,  4.5, alpha=0.05, color="green")
    ax_active.text(0.5, ax_active.get_ylim()[1] * 0.95, "COMPLEX", ha="center", color="red",   fontsize=8, alpha=0.7)
    ax_active.text(2.0, ax_active.get_ylim()[1] * 0.95, "MEDIUM",  ha="center", color="olive", fontsize=8, alpha=0.7)
    ax_active.text(3.5, ax_active.get_ylim()[1] * 0.95, "SIMPLE",  ha="center", color="green", fontsize=8, alpha=0.7)

    # ---- Difficulty trajectory ----
    ax_diff.set_title("Class Difficulty Trajectory\n(NCG-Selective)")
    ax_diff.set_xlabel("Epoch")
    ax_diff.set_ylabel("Difficulty Score")
    ncg_label = "NCG-Selective (dynamic)"
    ncg_epochs = all_data.get(ncg_label, {}).get("epochs", [])
    diff_colors = ["#e53935", "#d81b60", "#fdd835", "#43a047", "#1e88e5"]
    for c in range(N_CLASSES):
        xs_d = [e["epoch"] for e in ncg_epochs if "class_difficulty" in e]
        ys_d = [e["class_difficulty"][c] for e in ncg_epochs if "class_difficulty" in e]
        if xs_d:
            ax_diff.plot(xs_d, ys_d, color=diff_colors[c], linewidth=1.5,
                         label=CLASS_NAMES_SHORT[c].replace("\n", " "), marker=".", markersize=4)
    ax_diff.legend(fontsize=6)
    ax_diff.grid(True, alpha=0.3)

    out_path = os.path.join(SCRIPT_DIR, "comparison_softhebb.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")

    print("\n" + "=" * 60)
    print("  FINAL ACCURACY SUMMARY (SoftHebb)")
    print("=" * 60)
    for label, data in all_data.items():
        epochs = data["epochs"]
        if not epochs:
            print(f"  {label}: no data")
            continue
        final = epochs[-1]
        val  = final.get("val_acc",  float("nan"))
        test = final.get("test_acc", float("nan"))
        best_val  = max((e.get("val_acc",  0) for e in epochs), default=0)
        best_test = max((e.get("test_acc", 0) for e in epochs), default=0)
        print(f"  {label}")
        print(f"    Final  val={val:.4f}  test={test:.4f}")
        print(f"    Best   val={best_val:.4f}  test={best_test:.4f}")
        n_total = STATIC_NEURONS.get(label)
        apc = get_final_active(epochs, n_total)
        if apc:
            print(f"    Active neurons: {apc[:N_CLASSES]}  total={sum(apc[:N_CLASSES])}")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
