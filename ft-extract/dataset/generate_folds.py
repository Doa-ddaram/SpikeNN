import os
import shutil
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

"""
Code is adapted from https://gitlab.univ-lille.fr/fox/snn-pcn/
"""

def load_train_data(path, dtype_data=np.uint8, dtype_label=np.uint8):
    X = np.fromfile(path + "/X_train.bin", dtype=dtype_data)
    y = np.fromfile(path + "/y_train.bin", dtype=dtype_label)
    n_samples = len(y)
    X = X.reshape(n_samples,-1)
    return X,y


def save_data(path, X, y, set):
    with open(f"{path}/X_{set}.bin", "wb") as f:
        for x in X:
            f.write(x.ravel().tobytes())    
    with open(f"{path}/y_{set}.bin", "wb") as f:
        for label in y:
            f.write(label)


def generate_kfolds(data_dir, k):
    # Load data
    X, y = load_train_data(data_dir)
    
    # Make dir
    os.makedirs(f"{data_dir}/kfold/")

    # Split into folds
    kf = StratifiedKFold(n_splits=k, shuffle=False, random_state=None) 
    # iterate over the folds and save them into sub-directories
    for i, (train_index, val_index) in enumerate(kf.split(X, y)): # Training size is (k-1)/k
        # split X and y into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # Save new datasets into a sub-directory
        fold_dir = f"{data_dir}/kfold/{i}/"
        os.makedirs(fold_dir)
        save_data(fold_dir, X_train, y_train, "train")
        save_data(fold_dir, X_val, y_val, "val")
        # Copy test dataset
        shutil.copy(f"{data_dir}/X_test.bin", f"{fold_dir}/X_test.bin")
        shutil.copy(f"{data_dir}/y_test.bin", f"{fold_dir}/y_test.bin")



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory of the training data")
    parser.add_argument("--k", type=int, default=10, help="Number of folds")
    args = parser.parse_args()

    # Assert not already generated
    assert not os.path.isdir(f"{args.data_dir}/kfold/")

    generate_kfolds(args.data_dir, args.k)