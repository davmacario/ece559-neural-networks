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

CLASS_MAP = {
    "Circle": 0,
    "Heptagon": 1,
    "Hexagon": 2,
    "Nonagon": 3,
    "Octagon": 4,
    "Pentagon": 5,
    "Square": 6,
    "Star": 7,
    "Triangle": 8,
}


class MyNet(nn.Module):
    """
    MyNet
    ---
    Neural network used for shape classification.

    (from 0601-660603047-MACARIO.py)
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

        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 3)

        self.len_1st_fc = int(50 * 10 * 10)
        self.fc1 = nn.Linear(self.len_1st_fc, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc3 = nn.Linear(80, n_classes)

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
        2. Convolutional layer, 20 feature maps, 5x5 kernel, stride = 1,  -> (20 x 46 x 46)
        3. MaxPooling layer 2x2, stride = 2 - downsample by 2 -> (20 x 23 x 23)
        4. Convolutional layer, 50 feature maps, 3x3 kernel, stride = 1 -> (50 x 21 x 21)
        5. MaxPooling layer 2x2, stride = 2 - downsample by 2 -> (50 x 10 x 10)
        6. Flatten -> (1 x 5000)
        7. Fully connected layer, 5000 -> 150
        8. Fully connected layer, 150 -> 80
        9. Fully connected layer, 80 -> 9 - OUTPUT

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
        if VERB:
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

                print(
                    f"Epoch {epoch + 1}, Loss: {self.loss_train[epoch]}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Loss: {self.loss_train[epoch]}, Train Accuracy: {train_accuracy}%"
                )

        print("Finished Training!")
        torch.save(self.state_dict(), model_path)
        print("Model stored at {}".format(model_path))

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


def main():
    """main"""

    script_folder = os.path.dirname(__file__)
    model_path = os.path.join(script_folder, "0602-660603047-MACARIO_ubuntu.ZZZ")

    my_nn = MyNet()

    # Load the saved model state dictionary
    checkpoint = torch.load(model_path)

    # Load the model state dictionary into your model
    my_nn.load_state_dict(checkpoint["model_state_dict"])


if __name__ + +"__main__":
    main()
