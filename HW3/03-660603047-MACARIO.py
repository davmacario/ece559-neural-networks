import random
import numpy as np
import matplotlib.pyplot as plt
import os
import struct


def read_images(filename: str):
    with open(filename, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        data = data.reshape((size, nrows, ncols))
        # plt.imshow(data[0, :, :], cmap="gray")
        # plt.show()
    return data


def read_labels(filename: str):
    with open(filename, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        label = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        # print(label[0])
    return label


def step_function(x: float):
    return np.heaviside(x, 1)


def count_misclassifications(w, x, d):
    """
    count_misclassifications
    ---
    Count how many data set elements are not correctly classified
    by the current weights (w).
    """
    assert (
        len(d) == x.shape[1]
    ), f"The dimensions of x and d do not match ({len(d)} vs. {x.dim(1)})"
    d_prime = np.array([int(np.dot(x[:, i], w) >= 0) for i in range(x.shape[1])])
    return np.sum(np.absolute(d - d_prime))


def train(eta, eps, train_img, train_labels):
    W = np.random.normal(0, 1, (10, 784))
    epoch = 0

    for i in range(n):
        pass


if __name__ == "__main__":
    script_folder = os.path.dirname(__file__)
    imgpath = os.path.join(script_folder, "img")
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    ds_path = os.path.join(script_folder, "dataset")

    train_images = read_images(os.path.join(ds_path, "train-images-idx3-ubyte"))
    test_images = read_images(os.path.join(ds_path, "t10k-images-idx3-ubyte"))

    train_labels = read_labels(os.path.join(ds_path, "train-labels-idx1-ubyte"))
    test_labels = read_labels(os.path.join(ds_path, "t10k-labels-idx1-ubyte"))

    # Rows are vectors [0, ..., 0, 1, 0, ..., 0]
    train_vec = np.zeros((len(train_labels), 10))
    for i in range(len(train_labels)):
        # Create desired output vector
        train_vec[i, train_labels[i]] = 1

    test_vec = np.zeros((len(test_labels), 10))
    for i in range(len(test_labels)):
        test_vec[i, test_labels[i]] = 1

    ###
    n = 50000

    random.seed(660603047)
    np.random.seed(660603047)
