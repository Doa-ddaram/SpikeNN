"""
SoftHebb-based Merged CIFAR10 Dataset Preparation
==================================================
Applies merge map to pre-extracted SoftHebb features
(ft-extract/extracted/SoftHebb-CNN/CIFAR10/kfold/0-run1/)
and balances each merged class to equal sample count.

Feature dim: 24576 (SoftHebb 3-layer CNN output)
Original labels: 0-9 (CIFAR10)

Merging:
  Class 0 (COMPLEX, 3 merged): airplane(0) + automobile(1) + truck(9) -> 4500 train
  Class 1 (COMPLEX, 3 merged): bird(2) + deer(4) + horse(7)           -> 4500 train
  Class 2 (MEDIUM,  2 merged): cat(3) + dog(5)                        -> 4500 train
  Class 3 (SIMPLE,  1 class):  frog(6)                                -> 4500 train
  Class 4 (SIMPLE,  1 class):  ship(8)                                -> 4500 train
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

SOFTHEBB_DIR = os.path.join(REPO_ROOT, "ft-extract/extracted/SoftHebb-CNN/CIFAR10/kfold/0-run1")
OUT_DIR = os.path.join(os.path.dirname(__file__), "input-softhebb")

MERGE_MAP = {
    0: 0,  # airplane   -> complex_0
    1: 0,  # automobile -> complex_0
    9: 0,  # truck      -> complex_0
    2: 1,  # bird       -> complex_1
    4: 1,  # deer       -> complex_1
    7: 1,  # horse      -> complex_1
    3: 2,  # cat        -> medium_2
    5: 2,  # dog        -> medium_2
    6: 3,  # frog       -> simple_3
    8: 4,  # ship       -> simple_4
}

CLASS_NAMES = [
    "complex_0 (airplane + automobile + truck)",
    "complex_1 (bird + deer + horse)",
    "medium_2  (cat + dog)",
    "simple_3  (frog)",
    "simple_4  (ship)",
]

TARGET_PER_CLASS = {
    "train": 4500,
    "val":    500,
    "test":  1000,
}


def apply_merge(labels):
    return np.array([MERGE_MAP[int(l)] for l in labels], dtype=np.int16)


def subsample_balanced(data_list, labels, target_per_class, seed=42):
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > target_per_class:
            idx = rng.choice(idx, size=target_per_class, replace=False)
            idx = np.sort(idx)
        keep.append(idx)
    keep = np.concatenate(keep)
    keep = np.sort(keep)
    new_data = [data_list[i] for i in keep]
    new_labels = labels[keep]
    return new_data, new_labels


def process_split(split_name, target):
    src = os.path.join(SOFTHEBB_DIR, f"{split_name}.npy")
    raw = np.load(src, allow_pickle=True).item()

    data = raw["data"]
    labels = raw["labels"]
    shape = raw["shape"]
    max_t = raw.get("max_time", 1)

    merged_labels = apply_merge(labels)
    data_bal, labels_bal = subsample_balanced(data, merged_labels, target)

    dst = os.path.join(OUT_DIR, f"{split_name}.npy")
    np.save(dst, {
        "data":     data_bal,
        "labels":   labels_bal,
        "shape":    (len(data_bal), shape[1]),
        "max_time": max_t,
    })

    print(f"  {split_name}: {len(data_bal)} samples, shape dim={shape[1]}")
    for c in range(5):
        print(f"    [{c}] {CLASS_NAMES[c]}: {int((labels_bal == c).sum())}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Input: {SOFTHEBB_DIR}")
    print(f"Output: {OUT_DIR}\n")

    for split, target in TARGET_PER_CLASS.items():
        fname = "valset" if split == "val" else f"{split}set"
        print(f"Processing {split} (target {target}/class)...")
        process_split(fname, target)

    print("\nDone. Run the experiment with:")
    print("  python app/run.py experiments/cifar10-merged-classes/input-softhebb \\")
    print("    experiments/cifar10-merged-classes/output-softhebb-ncg \\")
    print("    experiments/cifar10-merged-classes/config-ncg-selective.json \\")
    print("    --seed 0")
