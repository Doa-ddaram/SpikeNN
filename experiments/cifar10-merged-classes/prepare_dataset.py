"""
Merged CIFAR10 Dataset Preparation
===================================
Purpose: Validate uncertainty-based dynamic neuron growth (NCG) by creating classes
         with different complexities (intra-class variance) through merging.

Merging strategy (CIFAR10 original class indices 0-9):
  Class 0 (COMPLEX, 3 merged): airplane(0) + automobile(1) + truck(9)
  Class 1 (COMPLEX, 3 merged): bird(2) + deer(4) + horse(7)
  Class 2 (MEDIUM,  2 merged): cat(3) + dog(5)
  Class 3 (SIMPLE,  1 class):  frog(6)
  Class 4 (SIMPLE,  1 class):  ship(8)

Expected result: NCG should allocate MORE active neurons to complex classes (0, 1)
                 due to their higher classification difficulty (intra-class variance).
"""

import os
import sys
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from spikenn.dataset import SpikingDataset

# CIFAR10 original label -> merged class mapping
# (0: airplane, 1: auto, 2: bird, 3: cat, 4: deer,
#  5: dog, 6: frog, 7: horse, 8: ship, 9: truck)
MERGE_MAP = {
    0: 0,  # airplane  -> complex_0
    1: 0,  # automobile-> complex_0
    9: 0,  # truck     -> complex_0
    2: 1,  # bird      -> complex_1
    4: 1,  # deer      -> complex_1
    7: 1,  # horse     -> complex_1
    3: 2,  # cat       -> medium_2
    5: 2,  # dog       -> medium_2
    6: 3,  # frog      -> simple_3
    8: 4,  # ship      -> simple_4
}

CLASS_NAMES = [
    "complex_0 (airplane + automobile + truck)",
    "complex_1 (bird + deer + horse)",
    "medium_2  (cat + dog)",
    "simple_3  (frog)",
    "simple_4  (ship)",
]

N_MERGED_CLASSES = 5


def apply_merge(y):
    return np.array([MERGE_MAP[int(label)] for label in y], dtype=np.int64)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "input")
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

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # TTFS coding: high intensity pixel -> early spike (timestamp = 1 - intensity)
    X_train_t = 1.0 - X_train
    X_val_t   = 1.0 - X_val
    X_test_t  = 1.0 - X_test

    SpikingDataset.from_numpy(X_train_t, y_train, max_time=1).save(f"{base}/trainset.npy")
    SpikingDataset.from_numpy(X_val_t,   y_val,   max_time=1).save(f"{base}/valset.npy")
    SpikingDataset.from_numpy(X_test_t,  y_test,  max_time=1).save(f"{base}/testset.npy")

    print(f"\nDataset saved to: {base}")
    print(f"  Train: {X_train_t.shape}  Val: {X_val_t.shape}  Test: {X_test_t.shape}")
    print(f"\nClass distribution (train, {N_MERGED_CLASSES} merged classes):")
    for c, name in enumerate(CLASS_NAMES):
        count = int((y_train == c).sum())
        print(f"  [{c}] {name}: {count} samples")
    print(f"\nTotal train: {len(y_train)}")
    print("\nDone. Now run: run_experiment.sh")
