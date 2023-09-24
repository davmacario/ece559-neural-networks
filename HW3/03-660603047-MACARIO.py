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


def count_misclassifications(W, x, d):
    """
    count_misclassifications
    ---
    Count how many data set elements are not correctly classified
    by the current weights (w).
    """
    assert (
        d.shape[0] == x.shape[0]
    ), f"The dimensions of x and d do not match ({d.shape[0]} vs. {x.shape[0]})"
    assert (
        d.shape[1] == W.shape[0]
    ), f"The dimensions of d and W don't match ({d.shape[1]} vs. {W.shape[1]})"

    n = x.shape[0]

    count_miss = 0

    # d_prime has shape (10 x n)
    for i in range(n):
        d_i_prime = np.dot(W, x[i])
        estim_i = np.argmax(d_i_prime)  # Estimated digit
        actual_i = np.argmax(d[i])
        if estim_i != actual_i:
            count_miss += 1

    return count_miss


def train(
    eta: float,
    eps: float,
    train_img,
    train_labels,
    max_iter=None,
    plots=False,
    imagepath="./img/",
):
    """
    train
    ---
    Train the single-layer neural network for digit classification.

    ### Input parameters:
    - eta: learning rate (> 0)
    - eps: tolerance on training (max. classification error allowed)
    - train_img: training images (shape: n x 28 x 28)
    - train_labels: training labels (associated with images) (shape:
    n x 10)
    - max_iter: if not None, it specifies the maximum number of
    iterations to be performed
    - plots: flag for printing plots (percentage of misclassifications
    over epochs)
    """

    assert eta > 0, "Eta must be a strictly positive value!"
    W = np.random.normal(0, 1, (10, 784))
    # Collapse images from 2D to 1D
    n = train_labels.shape[0]
    x = train_img.reshape((n, train_img.shape[1] * train_img.shape[2]))

    if max_iter is None:
        max_epoch = 1
    else:
        max_epoch = max_iter
    epoch = 0

    misses = (n + 1) * eps  # Initialize to start loop

    miss_in_time = []

    while misses / n > eps and epoch < max_epoch:
        # for i in range(n):
        #     v = np.dot(
        #         W,
        #     )
        misses = count_misclassifications(W, x, train_labels)
        miss_in_time.append(misses)
        print(f"Miss probability: {misses/n}")
        epoch += 1
        if max_iter is None:
            max_epoch += 1
        for i in range(n):
            x_i = x[i].reshape((x.shape[1], 1))
            W = W + eta * np.dot(
                train_labels[i].reshape((10, 1)) - step_function(np.dot(W, x_i)),
                x_i.T,
            )

    if plots:
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        ax.plot(list(range(epoch)), miss_in_time)
        ax.grid()
        plt.title(f"Number of misclassifications vs. epoch number, n = {n}")
        ax.set_xlabel(r"epoch")
        ax.set_ylabel(r"# misclassifications")
        plt.savefig(os.path.join(imagepath, f"miss_epoch_{n}-{eta}.png"))
        plt.show()

    return W


def test(W, test_img, test_labels):
    """
    test
    ---
    Perform inference with the evaluated weights.

    ### Input parameters
    - W: weight matrix (10 x 784)
    - test_img: array of test images (10000)
    - test_labels: labels associated to the test images
    """
    n = test_images.shape[0]
    assert (
        test_labels.shape[0] == n
    ), "The sizes of the test images and labels don't match"

    x = test_images.reshape((n, test_images.shape[1] * test_images.shape[2]))
    n_miss = count_misclassifications(W, x, test_labels)

    return n_miss


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
    print(train_images.shape)
    print(train_labels.shape)

    # Rows are vectors [0, ..., 0, 1, 0, ..., 0]
    train_vec = np.zeros((len(train_labels), 10))
    for i in range(len(train_labels)):
        # Create desired output vector
        train_vec[i, train_labels[i]] = 1

    test_vec = np.zeros((len(test_labels), 10))
    for i in range(len(test_labels)):
        test_vec[i, test_labels[i]] = 1

    ###
    n = 1000

    random.seed(660603047)
    np.random.seed(660603047)

    W = train(1, 0.01, train_images[:n], train_vec[:n], plots=True, imagepath=imgpath)

    print(
        f"Number of misclassifications on test set: {test(W, test_images, test_vec)} / {len(test_labels)}"
    )
