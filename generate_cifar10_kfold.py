import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms

def generate_cifar10_kfold(data_dir="cifar10_kfold", k=10):
    # --- Load CIFAR10 ---
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    # Convert to NumPy arrays
    X = np.stack([np.array(img) for img, _ in cifar_train])  # (50000, 3, 32, 32)
    y = np.array([label for _, label in cifar_train])        # (50000,)
    X_test = np.stack([np.array(img) for img, _ in cifar_test])  # (10000, 3, 32, 32)
    y_test = np.array([label for _, label in cifar_test])

    print(f"Train set: {X.shape}, labels: {y.shape}")
    print(f"Test set: {X_test.shape}, labels: {y_test.shape}")

    # --- Make directory ---
    os.makedirs(data_dir, exist_ok=True)

    # --- Stratified K-Fold ---
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_dir = os.path.join(data_dir, f"{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Save .npy files
        np.save(f"{fold_dir}/X_train.npy", X_train)
        np.save(f"{fold_dir}/y_train.npy", y_train)
        np.save(f"{fold_dir}/X_val.npy", X_val)
        np.save(f"{fold_dir}/y_val.npy", y_val)

        # Copy fixed test set to each fold
        np.save(f"{fold_dir}/X_test.npy", X_test)
        np.save(f"{fold_dir}/y_test.npy", y_test)

        print(f"✅ Fold {fold_idx} saved → train={len(train_idx)}, val={len(val_idx)}, test={len(y_test)}")

    print("\n🎯 CIFAR10 K-Fold dataset successfully generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CIFAR10 dataset in K-Fold format (fixed test set).")
    parser.add_argument("--data_dir", type=str, default="cifar10_kfold", help="Output directory for K-Fold data.")
    parser.add_argument("--k", type=int, default=10, help="Number of folds.")
    args = parser.parse_args()

    generate_cifar10_kfold(args.data_dir, args.k)