import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import shutil


def splitDataset(
    n_train: int,
    ds_path: str = os.path.join(os.path.dirname(__file__), "output"),
    shuffle: bool = True,
):
    """
    splitDataset
    ---
    Given the path of the dataset, sort the images into two folders 'train/'
    and 'test/' containing the same number of elements per class.

    The format of the folders is:

    `train/<class>/<image>.png`

    as stated in the PyTorch documentation.

    This function assumes that the file names are of the type:

    `<label>_<random numbers>.png`

    ### Input parameters
    - n_train: number of training elements for each class
    - ds_path: path of the data set (defaults to 'output' folder in same
    directory as the script)
    - shuffle: whether to shuffle the items before separation or not

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
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "train")):
        tr_path = os.path.join(os.path.dirname(__file__), "train")
        os.mkdir(tr_path)
    else:
        raise FileExistsError("'train' folder already exists")

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "test")):
        te_path = os.path.join(os.path.dirname(__file__), "test")
        os.mkdir(te_path)
    else:
        raise FileExistsError("'test' folder already exists")

    img_names = os.listdir(ds_path)  # File names (not abs. paths)
    class_labels = {}  # Key: class name, value: n. of items encountered so far

    # Shuffle items, if required
    if shuffle:
        random.shuffle(img_names)

    for fname in img_names:
        # Get class - if new one, append it to the vector of classes
        class_curr = str(fname.split("_")[0])
        if class_curr not in class_labels:
            class_labels[class_curr] = 0
            os.mkdir(os.path.join(tr_path, class_curr))
            os.mkdir(os.path.join(te_path, class_curr))
        else:
            class_labels[class_curr] += 1

        if class_labels[class_curr] <= n_train:
            # Place current image in training set
            shutil.copy(
                os.path.join(ds_path, fname), os.path.join(tr_path, class_curr, fname)
            )
        else:
            # Place current image in test set
            shutil.copy(
                os.path.join(ds_path, fname), os.path.join(te_path, class_curr, fname)
            )

    return list(class_labels.keys()), tr_path, te_path


def importDataset(
    train_path: str, classes: list = None, batch_size: int = 16, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    importTraining
    ---
    Import the training set, given the path.

    ### Input parameters
    - train_path: path of the trainin set folder
    - classes: list of classes - if not given, they will be inferred
    - batch_size: batch size in data loader
    - shuffle: flag to select whether to shuffle the training elements
    or not in data loader

    ### Output parameters
    - train_data_loader: `torch.utils.data.DataLoader` object containing the training set
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The specified path ({train_path}) is invalid!")

    img_names = os.listdir(train_path)
    train_labels = []
    for fname in img_names:
        train_labels.append(str(fname.split("_")[0]))

    transf = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
    # TODO: maybe add transform.Normalize()

    train_images = datasets.ImageFolder(train_path, transf)
    train_data_loader = torch.utils.data.DataLoader(
        train_images, batch_size=batch_size, shuffle=shuffle
    )

    return train_data_loader


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "output")
    train_path = os.path.join(os.path.dirname(__file__), "train")
    n_training = 8000

    try:
        splitDataset(n_training, dataset_path)
        print("Training and test sets generated!")
    except FileExistsError:
        print("Training and test set already split!")

    dl_train = importDataset(train_path)
