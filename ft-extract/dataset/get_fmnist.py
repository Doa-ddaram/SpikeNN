import os
import gzip
import requests
import numpy as np
from generate_train_val import generate_train_val
from generate_folds import generate_kfolds

"""
Code is adapted from https://gitlab.univ-lille.fr/fox/snn-pcn/
"""

def load_data(kind='train'):
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    labels_filename = '%s-labels-idx1-ubyte.gz' % kind
    images_filename = '%s-images-idx3-ubyte.gz' % kind

    # Download labels file
    labels_url = base_url + labels_filename
    response = requests.get(labels_url)
    labels = np.frombuffer(gzip.decompress(response.content), dtype=np.uint8, offset=8)

    # Download images file
    images_url = base_url + images_filename
    response = requests.get(images_url)
    images = np.frombuffer(gzip.decompress(response.content), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":

    X_train, y_train = load_data(kind='train')
    X_test, y_test = load_data(kind='t10k')

    os.makedirs("F-MNIST/")

    with open(f'F-MNIST/X_train.bin', 'wb') as f:
        for x in X_train:
            f.write(x.ravel().tobytes())
    with open(f'F-MNIST/X_test.bin', 'wb') as f:
        for x in X_test:
            f.write(x.ravel().tobytes())
    with open(f"F-MNIST/y_train.bin", "wb") as f:
        for label in y_train:
            f.write(label)
    with open(f"F-MNIST/y_test.bin", "wb") as f:
        for label in y_test:
            f.write(label)
            
    generate_train_val("F-MNIST/", 0.1, 0)
    generate_kfolds("F-MNIST/", 10)