import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import shutil


def splitDataset(
    n_train: int, ds_path: str = os.path.join(os.path.dirname(__file__), "output")
):
    """
    splitDataset
    ---
    Given the path of the dataset, sort the images into two folders 'train/'
    and 'test/' containing the same number of elements per class.

    This function assumes that the file names are of the type:

    `<label>_<random numbers>.png`

    ### Input parameters
    - n_train: number of training elements for each class
    - ds_path: path of the data set (defaults to 'output' folder in same
    directory as the script)

    ### Output parameters
    - class_labels: list of class labels extracted from the file names
    - tr_path: folder of training images (abs. path)
    - te_path: folder of test images (abs. path)
    """
    assert os.path.isdir(
        ds_path
    ), "The provided path ({}) is not a valid directory!".format(ds_path)

    # Create 'train' and 'test' folder (same dir as python file)
    # If either already exists, an exception is raised!
    if not os.path.exists(os.path.join(__file__, "train")):
        tr_path = os.path.join(__file__, "train")
        os.mkdir(tr_path)
    else:
        raise FileExistsError("'train' folder already exists")

    if not os.path.exists(os.path.join(__file__, "test")):
        te_path = os.path.join(__file__, "test")
        os.mkdir(te_path)
    else:
        raise FileExistsError("'test' folder already exists")

    img_names = os.listdir(ds_path)  # File names (not abs. paths)
    class_labels = {}  # Key: class name, value: n. of items encountered so far

    # TODO: maybe shuffle image names (but classes vector will have a different order)

    for fname in img_names:
        # Get class - if new one, append it to the vector of classes
        class_curr = str(fname.split("_")[0])
        if class_curr not in class_labels:
            class_labels[class_curr] = 0
        else:
            class_labels[class_curr] += 1

        if class_labels[class_curr] <= n_train:
            # Place current image in training set
            shutil.copy(os.path.join(ds_path, fname), os.path.join(tr_path, fname))
        else:
            # Place current image in test set
            shutil.copy(os.path.join(ds_path, fname), os.path.join(te_path, fname))

    return list(class_labels.keys()), tr_path, te_path


def importDataset(train_path: str, test_path: str, classes: list):
    """
    importDataset
    ---
    """


if __name__ == "__main__":
    pass
