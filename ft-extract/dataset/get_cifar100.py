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
    filename = 'cifar-100-python.tar.gz'

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
    data_dir = 'cifar-100-python/'

    with open('cifar-100-python/train', 'rb') as file:
        train_data = pickle.load(file, encoding='bytes')
    X_train = np.array(train_data[b'data']).reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).reshape(50000, -1)
    y_train = np.array(train_data[b'fine_labels'], dtype=np.uint8)
    
    with open('cifar-100-python/test', 'rb') as file:
        test_data = pickle.load(file, encoding='bytes')
    X_test = np.array(test_data[b'data']).reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).reshape(10000, -1)
    y_test = np.array(test_data[b'fine_labels'], dtype=np.uint8)

    shutil.rmtree(data_dir)

    os.makedirs("CIFAR100/")

    with open(f'CIFAR100/X_train.bin', 'wb') as f:
        for x in X_train:
            f.write(x.ravel().tobytes())
    with open(f'CIFAR100/X_test.bin', 'wb') as f:
        for x in X_test:
            f.write(x.ravel().tobytes())
    with open(f"CIFAR100/y_train.bin", "wb") as f:
        for label in y_train:
            f.write(label)
    with open(f"CIFAR100/y_test.bin", "wb") as f:
        for label in y_test:
            f.write(label)
            
    generate_train_val("CIFAR100/", 0.1, 0)
    generate_kfolds("CIFAR100/", 10)