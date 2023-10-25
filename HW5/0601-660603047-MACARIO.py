#!/usr/bin/env python3

import torch
from torch import nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import shutil
from typing import Callable

DEBUG = False
VERB = True
PLOTS = False
MPS = True
CUDA = True


class MyNet(nn.Module):
    """
    MyNet
    ---
    Neural network used for shape classification.
    """

    def __init__(
        self,
        input_shape: (int, int),
        n_classes: int,
        act_function: Callable = nn.functional.relu,
    ):
        """
        Initialize the neural network.

        ### Input parameters
        - input_shape: tuple of integers indicating the shape of the input
        (maps x height x width)
        - n_classes: number of classes for the classification model
        - act_function: activation function of layers (not last one)

        ### Network structure (layers)
        -
        """
        super(MyNet, self).__init__()

        self.input_size = input_shape
        self.n_classes = n_classes

        # Activation function
        self.act_func = act_function

        # Layer Definition
        self.pool_halve = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 15, 5)
        self.conv2 = nn.Conv2d(15, 30, 5)

        self.len_1st_fc = int(30 * 9 * 9)
        self.fc1 = nn.Linear(self.len_1st_fc, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, n_classes)

        # Variables for displaying performance
        self.n_epochs = None
        self.loss_train = None
        self.loss_test = None
        self.acc_train = None
        self.acc_test = None

    def forward(self, x, softmax_out: bool = False):
        """
        forward
        ---
        Forward propagation in the neural network.

        ### Network structure

        All convolutional and fully connected layers excluding the output layer use the
        activation function passed to the class constructor.

        1. MaxPooling layer, 4x4, stride = 4 - downsampling images by 4 -> (3 x 50 x 50)
        2. Convolutional layer, 15 feature maps, 5x5 kernel, stride = 1,  -> (15 x 46 x 46)
        3. MaxPooling layer 2x2, stride = 2 - downsample by 2 -> (15 x 23 x 23)
        4. Convolutional layer, 30 feature maps, 5x5 kernel, stride = 1 -> (30 x 19 x 19)
        5. MaxPooling layer 2x2, stride = 2 - downsample by 2 -> (30 x 9 x 9)
        6. Flatten -> (1 x 5000)
        7. Fully connected layer, 5000 -> 120
        8. Fully connected layer, 120 -> 60
        9. Fully connected layer, 60 -> 9 - OUTPUT

        ---

        ### Input parameters
        - x: input of the network
        - softmax_out: flag to indicate whether to apply softmax function at the
        output of the network

        ### Output parameters
        - y: output of the network [array]
        """
        y = self.pool_4(x)
        y = self.pool_halve(self.act_func(self.conv1(y)))
        y = self.pool_halve(self.act_func(self.conv2(y)))
        if DEBUG:
            print(y.shape)
        y = y.view(-1, self.len_1st_fc)
        y = self.act_func(self.fc1(y))
        y = self.act_func(self.fc2(y))
        if softmax_out:
            y = nn.functional.softmax(self.fc3(y), -1)
        else:
            y = self.fc3(y)
        return y

    def train_nn(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Callable,
        obj_func: Callable,
        n_epochs: int,
        test_dataloader: torch.utils.data.DataLoader = None,
        model_path: str = "shapes_model.pth",
        _device=None,
    ):
        """
        train_nn
        ---
        Train the neural network.

        ### Input parameters
        - train_dataloader: dataloader containing training data
        - optimizer: training method
        - obj_function: function to be minimized by the training procedure
        - n_epochs: number of training epochs
        - model_path: path of the output model
        - mps_device: if passed, specify the presence of MPS (Apple silicon GPU)
        """
        self.n_epochs = n_epochs
        self.loss_train = np.zeros((n_epochs,))
        self.acc_train = np.zeros((n_epochs,))

        if test_dataloader is not None:
            self.loss_test = np.zeros((n_epochs,))
            self.acc_test = np.zeros((n_epochs,))

        for epoch in range(n_epochs):  # Change the number of epochs as needed
            running_loss = 0.0
            # print(loadingBar(epoch, n_epochs, 20), end="\r")
            for i, data in enumerate(train_dataloader, 0):
                if VERB:
                    print(
                        f"> Current epoch - {loadingBar(i, len(train_dataloader), 30)} {round(100 * i / len(train_dataloader), 3)}%",
                        end="\r",
                    )

                # Take the current batch and separate the image (inputs) and the labels
                inputs, labels = data

                if _device is not None:
                    # If using GPU, move data to the device
                    inputs = inputs.to(_device)
                    labels = labels.to(_device)

                # Clear the values of the gradients (from prev. iteration)
                optimizer.zero_grad()

                # Extract the output to the current inputs (batch)
                outputs = self(inputs)
                # Evaluate the loss (plug actual labels vs. estimated outputs in obj func.)
                loss = obj_func(outputs, labels)

                # Backpropagation
                loss.backward()
                # Update model parameters
                optimizer.step()

                running_loss += loss.item()

            if VERB:
                print(
                    f"Epoch {epoch + 1} done!                                                        ",
                    end="\r",
                )

            # Store results
            train_accuracy = eval_accuracy(self, train_dataloader, _device)
            self.loss_train[epoch] = running_loss / len(train_dataloader)
            self.acc_train[epoch] = train_accuracy

            if test_dataloader is not None:
                test_accuracy = eval_accuracy(self, test_dataloader, _device)
                self.acc_test[epoch] = test_accuracy

                loss_te = 0.0
                for i, data_te in enumerate(test_dataloader, 0):
                    in_te, lab_te = data_te

                    if _device is not None:
                        in_te = in_te.to(_device)
                        lab_te = lab_te.to(_device)

                    loss_te += obj_func(self(in_te), lab_te).item()
                self.loss_test[epoch] = loss_te / len(train_dataloader)

                if epoch == 0:
                    min_loss_test = self.loss_test[epoch]
                    best_epoch = epoch

                if self.loss_test[epoch] <= min_loss_test:
                    # The saved model contains the parameters that perform best over the whole training
                    torch.save(self.state_dict(), model_path)

                if VERB:
                    print(
                        f"Epoch {epoch + 1}, Train Loss: {round(self.loss_train[epoch], 4)}, Train Accuracy: {round(train_accuracy, 4)}%, Test Loss: {round(self.loss_test[epoch], 4)}, Test Accuracy: {round(test_accuracy, 4)}"
                    )
            else:
                if VERB:
                    print(
                        f"Epoch {epoch + 1}, Loss: {self.loss_train[epoch]}, Train Accuracy: {train_accuracy}%"
                    )

        if VERB:
            print("Finished Training!")
            print(f"Model stored at {model_path} - from epoch {best_epoch}")

    def print_results(self, flg_te: bool = False, out_folder: str = None):
        """
        print_results
        ---
        Display plots of loss and accuracy vs. epoch.

        ### Input parameters
        - flg_te: flag indicating whether to include the test set measurements
        - out_folder: destination of produced plots
        """
        if not os.path.isdir(out_folder):
            raise NotADirectoryError("The specified path is not a directory!")

        if self.loss_train is None or self.acc_train is None:
            raise ValueError("Model was not trained!")

        if flg_te and self.loss_test is None:
            raise ValueError("Missing test (validation) set evaluation!")

        # Plot loss vs. epoch
        plt.figure(figsize=(10, 8))
        plt.plot(
            list(range(1, self.n_epochs + 1)), self.loss_train, "b", label="Train loss"
        )
        if flg_te:
            plt.plot(
                list(range(1, self.n_epochs + 1)),
                self.loss_test,
                "r",
                label="Test loss",
            )
            plt.legend()
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. epoch number")
        plt.tight_layout()
        if out_folder is not None:
            plt.savefig(os.path.join(out_folder, "loss_vs_epoch.png"))
        plt.show()

        # Plot accuracy vs. epoch
        plt.figure(figsize=(10, 8))
        plt.plot(
            list(range(1, self.n_epochs + 1)),
            self.acc_train,
            "b",
            label="Train accuracy",
        )
        if flg_te:
            plt.plot(
                list(range(1, self.n_epochs + 1)),
                self.acc_test,
                "r",
                label="Test accuracy",
            )
            plt.legend()
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs. epoch number")
        plt.tight_layout()
        if out_folder is not None:
            plt.savefig(os.path.join(out_folder, "acc_vs_epoch.png"))
        plt.show()


# +--------------------------------------------------------------------------------------+


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
            class_labels[class_curr] = 1
            os.mkdir(os.path.join(tr_path, class_curr))
            os.mkdir(os.path.join(te_path, class_curr))
        else:
            class_labels[class_curr] += 1

        if class_labels[class_curr] <= n_train:
            # Place current image in training set
            shutil.copy(
                os.path.join(ds_path, fname), os.path.join(
                    tr_path, class_curr, fname)
            )
        else:
            # Place current image in test set
            shutil.copy(
                os.path.join(ds_path, fname), os.path.join(
                    te_path, class_curr, fname)
            )

    return list(class_labels.keys()), tr_path, te_path


def importDataset(
    ds_path: str, batch_size: int = 32, shuffle: bool = True
) -> (torch.utils.data.DataLoader, dict):
    """
    importDataset
    ---
    Import the training set, given the path.

    ### Input parameters
    - ds_path: path of the trainin set folder
    - batch_size: batch size in data loader
    - shuffle: flag to select whether to shuffle the training elements
    or not in data loader

    ### Output parameters
    - train_data_loader: `torch.utils.data.DataLoader` object containing the training set
    - label_class_mapping: mapping between labels (integer numbers) and classes (dict)
    """
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"The specified path ({ds_path}) is invalid!")

    img_names = os.listdir(ds_path)
    train_labels = []
    for fname in img_names:
        train_labels.append(str(fname.split("_")[0]))

    transf = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds_images = datasets.ImageFolder(ds_path, transf)
    ds_data_loader = torch.utils.data.DataLoader(
        ds_images, batch_size=batch_size, shuffle=shuffle
    )

    label_class_mapping = ds_images.class_to_idx

    return ds_data_loader, label_class_mapping


def dispImages(
    images: torch.tensor,
    labels: torch.tensor,
    classes_map: dict,
    img_path: str = None,
    plot_shape: tuple = (2, 3),
):
    """
    dispImages
    ---
    Display a subset of the dataset images.

    ### Input parameters
    - images: tensor of images
    - labels: tensor of labels associated with the images
    - classes_map: dictionary containing the mapping between labels
    and classes
    - img_path: path of the output image; if None, the image is not
    saved
    - plot_shape: optional tuple indicating the 'shape' of the plot
    in terms of subplots
    """
    assert isinstance(plot_shape[0], int) and isinstance(
        plot_shape[1], int
    ), "The elements of 'plot_shape' should be integers!"
    plt.figure(figsize=(10, 5))
    for i in range(plot_shape[0] * plot_shape[1]):
        plt.subplot(plot_shape[0], plot_shape[1], i + 1)
        plt.imshow(
            np.transpose(images[i], (1, 2, 0)) / 2 + 0.5
        )  # Unnormalize and display the image
        plt.title(list(classes_map.keys())[labels[i]])
        plt.axis("off")
        if img_path is not None:
            plt.savefig(img_path)
    plt.show()


def eval_accuracy(
    network: MyNet, data_loader: torch.utils.data.DataLoader, _device=None
):
    """
    eval_accuracy
    ---
    Return the accuracy of the neural network on input data set.

    ### Input parameters
    - network: tested neural network (MyNet object)
    - data_loader: data set on which to evaluate accuracy
    - mps_dev: if not None, specify MPS device to be used

    ### Output parameter
    - Accuracy (in percentage)
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            # Iterate on batches
            inputs, labels = data
            if _device is not None:
                # Use GPU if available
                inputs = inputs.to(_device)
                labels = labels.to(_device)
            outputs = network(inputs)
            # The output is evaluated on current *batch*!
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def loadingBar(
    current_iter: int, tot_iter: int, n_chars: int = 10, ch: str = "=", n_ch: str = " "
) -> str:
    """
    loadingBar
    ---
    Produce a loading bar string to be printed.

    ### Input parameters
    - current_iter: current iteration, will determine the position
    of the current bar
    - tot_iter: total number of iterations to be performed
    - n_chars: total length of the loading bar in characters
    - ch: character that makes up the loading bar (default: =)
    - n_ch: character that makes up the remaining part of the bar
    (default: blankspace)
    """
    n_elem = int(current_iter * n_chars / tot_iter)
    prog = str("".join([ch] * n_elem))
    n_prog = str("".join([n_ch] * (n_chars - n_elem - 1)))
    return "[" + prog + n_prog + "]"


# +--------------------------------------------------------------------------------------+


def main():
    script_folder = os.path.dirname(__file__)
    dataset_path = os.path.join(script_folder, "output")
    train_path = os.path.join(script_folder, "train")
    test_path = os.path.join(script_folder, "test")
    images_folder = os.path.join(script_folder, "img")
    model_path = os.path.join(script_folder, "0602-660603047-MACARIO_mac.ZZZ")
    n_training = 8000

    try:
        splitDataset(n_training, dataset_path)
        print("Training and test sets generated!")
    except FileExistsError:
        print("Training and test set already split!")

    dl_train, classes_map = importDataset(train_path)
    dl_test, _ = importDataset(test_path)

    tr_img, tr_labels = next(iter(dl_train))

    if DEBUG:
        print(tr_img)
        print()
        print(tr_labels)

    if PLOTS:
        dispImages(
            tr_img,
            tr_labels,
            classes_map,
            img_path=os.path.join(images_folder, "train_samples.png"),
        )

    if DEBUG:
        print(tr_img.shape[2:4])

    # Define Neural Network
    my_nn = MyNet(tr_img.shape[2:4], len(classes_map.keys()))

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

    # Launch training

    if torch.backends.mps.is_available() and MPS:
        print("Using MPS!")
        mps_device = torch.device("mps")
        my_nn.to(mps_device)
        model_path = os.path.join(
            script_folder, "0602-660603047-MACARIO_mac.ZZZ")
        my_nn.train_nn(
            dl_train, optimizer, criterion, 20, dl_test, model_path, mps_device
        )
    elif torch.cuda.is_available() and CUDA:
        print("Using CUDA!")
        cuda_device = torch.device("cuda")
        my_nn.to(cuda_device)
        model_path = os.path.join(
            script_folder, "0602-660603047-MACARIO_ubuntu_A.ZZZ")
        my_nn.train_nn(
            dl_train, optimizer, criterion, 10, dl_test, model_path, cuda_device
        )
    else:
        model_path = os.path.join(
            script_folder, "0602-660603047-MACARIO_cpu.ZZZ")
        my_nn.train_nn(dl_train, optimizer, criterion, 10, dl_test, model_path)

    # Print results
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
    my_nn.print_results(flg_te=True, out_folder=images_folder)


if __name__ == "__main__":
    main()
