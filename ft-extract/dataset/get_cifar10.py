import os
import requests
import numpy as np
import pickle
import shutil
import tarfile
from generate_train_val import generate_train_val
from generate_folds import generate_kfolds

"""
Code is adapted from https://gitlab.univ-lille.fr/fox/snn-pcn/
"""

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    x = dict[b"data"]
    y = dict[b"labels"]
    # Reshape and invert channel dimension
    x = x.reshape(x.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).reshape(x.shape[0], -1)
    y = np.array(y, dtype=np.uint8)
    return x, y


def download_dataset():
    base_url = 'https://www.cs.toronto.edu/~kriz/'
    filename = 'cifar-10-python.tar.gz'

    # Download CIFAR-10 dataset
    url = base_url + filename
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

    # Extract CIFAR-10 dataset
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()

    os.remove(filename)


if __name__ == "__main__":
    download_dataset()
    data_dir = 'cifar-10-batches-py/'

    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = data_dir + "data_batch_{}".format(i)
        x, y = load_batch(file_path)
        X_train.append(x)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_batch(data_dir + "test_batch")

    shutil.rmtree(data_dir)

    os.makedirs("CIFAR10/")

    with open(f'CIFAR10/X_train.bin', 'wb') as f:
        for x in X_train:
            f.write(x.ravel().tobytes())
    with open(f'CIFAR10/X_test.bin', 'wb') as f:
        for x in X_test:
            f.write(x.ravel().tobytes())
    with open(f"CIFAR10/y_train.bin", "wb") as f:
        for label in y_train:
            f.write(label)
    with open(f"CIFAR10/y_test.bin", "wb") as f:
        for label in y_test:
            f.write(label)
            
    generate_train_val("CIFAR10/", 0.1, 0)
    generate_kfolds("CIFAR10/", 10)