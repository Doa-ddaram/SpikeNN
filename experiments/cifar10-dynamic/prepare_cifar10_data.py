import os
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from spikenn.dataset import SpikingDataset

base = "experiments/cifar10-dynamic/input"
os.makedirs(base, exist_ok=True)

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

X_train = np.stack([np.array(img, dtype=np.float32) / 255.0 for img, _ in train])
y_train = np.array([y for _, y in train])
X_test = np.stack([np.array(img, dtype=np.float32) / 255.0 for img, _ in test])
y_test = np.array([y for _, y in test])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# Timestamp coding in [0,1]: higher intensity -> earlier spike.
X_train_t = 1.0 - X_train
X_val_t = 1.0 - X_val
X_test_t = 1.0 - X_test

SpikingDataset.from_numpy(X_train_t, y_train, max_time=1).save(f"{base}/trainset.npy")
SpikingDataset.from_numpy(X_val_t, y_val, max_time=1).save(f"{base}/valset.npy")
SpikingDataset.from_numpy(X_test_t, y_test, max_time=1).save(f"{base}/testset.npy")

print("saved:", base)
print("train:", X_train_t.shape, "val:", X_val_t.shape, "test:", X_test_t.shape)
