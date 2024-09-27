import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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


def split_train_val(X, y, validation_size=0.1, seed=0):
    if validation_size <= 0: return X, y, [], []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    for train_index, val_index in sss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    return X_train, y_train, X_val, y_val 


def generate_train_val(data_dir, validation_size, seed):
    out_dir = data_dir + "/gs/"
    
    # Load and split data
    X, y = load_train_data(data_dir)
    X_train, y_train, X_val, y_val = split_train_val(X, y, validation_size, seed)
    
    # Create directory
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    
    # Save data
    save_data(out_dir, X_train, y_train, "train")
    save_data(out_dir, X_val, y_val, "val")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory of the training data")
    parser.add_argument("--validation_size", type=int, default=0.1, help="Validation size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    generate_train_val(args.data_dir, args.validation_size, args.seed)