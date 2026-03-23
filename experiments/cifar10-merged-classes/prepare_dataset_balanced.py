"""
Balanced Merged CIFAR10 Dataset Preparation
=============================================
Same merging strategy as prepare_dataset.py, but each merged class is
subsampled to 4,500 train samples (= same as the smallest single class).

This eliminates the sample-count advantage that confounded the first experiment,
so that intra-class variance alone drives difficulty differences.

Merging:
  Class 0 (COMPLEX, 3 merged): airplane(0) + automobile(1) + truck(9)  -> 4500 train (1500 each)
  Class 1 (COMPLEX, 3 merged): bird(2) + deer(4) + horse(7)            -> 4500 train (1500 each)
  Class 2 (MEDIUM,  2 merged): cat(3) + dog(5)                         -> 4500 train (2250 each)
  Class 3 (SIMPLE,  1 class):  frog(6)                                 -> 4500 train
  Class 4 (SIMPLE,  1 class):  ship(8)                                 -> 4500 train

Expected: class 0 and 1 have higher error because their intra-class variance
          forces neurons to represent 3 distinct visual patterns.
          NCG should allocate more active neurons to classes 0 and 1.
"""

import os
import sys
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from spikenn.dataset import SpikingDataset

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

# Target samples per merged class in TRAIN set (before val split)
# CIFAR10 has 5000 per class → after 0.9 train split ≈ 4500 per class
TARGET_PER_CLASS = 4500


def apply_merge(y):
    return np.array([MERGE_MAP[int(label)] for label in y], dtype=np.int64)


def subsample_balanced(X, y, target_per_class, seed=42):
    """Keep at most target_per_class samples per class, stratified."""
    rng = np.random.RandomState(seed)
    keep_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > target_per_class:
            idx = rng.choice(idx, size=target_per_class, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    keep_idx = np.sort(keep_idx)
    return X[keep_idx], y[keep_idx]


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "input-balanced")
    os.makedirs(base, exist_ok=True)

    print("Loading CIFAR10...")
    train = datasets.CIFAR10(root=os.path.join(REPO_ROOT, "data"), train=True,  download=True, transform=transforms.ToTensor())
    test  = datasets.CIFAR10(root=os.path.join(REPO_ROOT, "data"), train=False, download=True, transform=transforms.ToTensor())

    X_train = np.stack([np.array(img, dtype=np.float32) / 255.0 for img, _ in train])
    y_train = np.array([y for _, y in train])
    X_test  = np.stack([np.array(img, dtype=np.float32) / 255.0 for img, _ in test])
    y_test  = np.array([y for _, y in test])

    # Apply class merging
    y_train = apply_merge(y_train)
    y_test  = apply_merge(y_test)

    # Subsample to balance
    print(f"\nSubsampling train to {TARGET_PER_CLASS} samples per merged class...")
    X_train, y_train = subsample_balanced(X_train, y_train, TARGET_PER_CLASS, seed=42)

    # Also balance test set (1000 per class = CIFAR10 default for single classes)
    X_test, y_test = subsample_balanced(X_test, y_test, 1000, seed=42)

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # TTFS coding
    X_train_t = 1.0 - X_train
    X_val_t   = 1.0 - X_val
    X_test_t  = 1.0 - X_test

    SpikingDataset.from_numpy(X_train_t, y_train, max_time=1).save(f"{base}/trainset.npy")
    SpikingDataset.from_numpy(X_val_t,   y_val,   max_time=1).save(f"{base}/valset.npy")
    SpikingDataset.from_numpy(X_test_t,  y_test,  max_time=1).save(f"{base}/testset.npy")

    print(f"\nDataset saved to: {base}")
    print(f"  Train: {X_train_t.shape}  Val: {X_val_t.shape}  Test: {X_test_t.shape}")
    print(f"\nClass distribution (train, balanced):")
    for c, name in enumerate(CLASS_NAMES):
        count = int((y_train == c).sum())
        print(f"  [{c}] {name}: {count} samples")
    print(f"\nTotal train: {len(y_train)}")
    print("\nDone. Run: run_experiment_balanced.sh")
