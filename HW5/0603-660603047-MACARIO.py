#!/usr/bin/env python3

import torch
from torch import nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
import random
import shutil
from typing import Callable

MPS = True
CUDA = True
VERB = True
DEBUG = False
IMG_SHAPE = (3, 200, 200)
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
    """

    def __init__(
        self,
        input_shape: (int, int),
        class_map: dict,
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
        self.n_classes = len(class_map.keys())
        self.class_mapping = class_map

        # Activation function
        self.act_func = act_function

        # Layer Definition
        self.pool_halve = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.conv3 = nn.Conv2d(50, 70, 3)

        self.len_1st_fc = int(70 * 4 * 4)
        self.fc1 = nn.Linear(self.len_1st_fc, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, self.n_classes)

        self.do = nn.Dropout(0.2)

        # Variables for displaying performance
        self.n_epochs = None
        self.loss_train = None
        self.loss_test = None
        self.acc_train = None
        self.acc_test = None

    def forward(self, x, softmax_out: bool = False, dropout: bool = True):
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
        6. Convolutional layer, 70 feature maps, 3x3 kernel, stride = 1 -> (70 x 8 x 8)
        7. MaxPooling layer 2x2, stride = 2 - downsample by 2 -> (70 x 4 x 4)
        8. Flatten -> (1 x 70*16)
        9. Fully connected layer, 70*16 -> 120
        10. Fully connected layer, 120 -> 60
        11. Fully connected layer, 60 -> 9 - OUTPUT

        ---

        ### Input parameters
        - x: input of the network
        - softmax_out: flag to indicate whether to apply softmax function at the
        output of the network
        - dropout: flag to select dropout

        ### Output parameters
        - y: output of the network [array]
        """
        y = self.pool_4(x)
        y = self.pool_halve(self.act_func(self.conv1(y)))
        y = self.pool_halve(self.act_func(self.conv2(y)))
        y = self.pool_halve(self.act_func(self.conv3(y)))
        if DEBUG:
            print(y.shape)
        y = y.view(-1, self.len_1st_fc)
        if dropout:
            y = self.do(y)
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
        - test_dataloader: if not None, it is the available device that can be used for training (GPU)
        - model_path: path of the output model
        - _device: if passed, specify the presence of MPS (Apple silicon GPU)
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
                    max_acc_test = self.acc_test[epoch]
                    best_epoch = epoch

                if self.acc_test[epoch] >= max_acc_test:
                    # The saved model contains the parameters that perform best over
                    #  the whole training in terms of accuracy on the validation set
                    torch.save(self.state_dict(), model_path)
                    max_acc_test = self.acc_test[epoch]
                    best_epoch = epoch

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
            print(f"Model stored at {model_path} - from epoch {best_epoch + 1}")

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

    def load_parameters(self, model_path: str, _device=None) -> int:
        """
        load_parameters
        ---
        Load model parameters from a file, given its path.

        ### Input parameters
        - model_path: path of the trained model parameters

        ### Output parameter
        - 1: if success
        """
        # Open file
        # if _device is not None:
        #    checkpoint = torch.load(model_path, map_location=_device)
        # else:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        # Load parameters
        self.load_state_dict(checkpoint)
        # Switch to evaluation mode
        self.eval()

        self.net_loaded = True

        return 1

    def inference(
        self, image_path: str, plot: bool = False, img_path: str = None, _device=None
    ) -> str:
        """
        inference
        ---
        Evaluate the NN output given an image (from path).

        ### Input parameters
        - image_path: path of the image to be classified
        - plot: flag to display the plot
        - img_path: if set, save image here
        - _device: if not None, indicates the device on which to perform inference (GPU)

        ### Output parameters
        - label: label (string), mapped
        """
        if not self.net_loaded:
            raise ValueError("The network is not in evaluation mode!")

        transf = transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        image = Image.open(image_path).convert("RGB")

        in_tensor = transf(image)
        in_batch = in_tensor.unsqueeze(0)

        if _device is not None:
            in_batch = in_batch.to(_device)

        # Pass through network & get index
        with torch.no_grad():
            output = self(in_batch)
            _, pred_idx = torch.max(output, 1)

        # Map index to label
        pred_label = list(self.class_mapping.keys())[pred_idx]

        if plot:
            plt.figure(figsize=(10, 6))
            plt.imshow(np.transpose(in_tensor, (1, 2, 0)) / 2 + 0.5)
            plt.title(f"Estimated class: {pred_label}")
            plt.tight_layout()
            if img_path is not None:
                plt.savefig(img_path)
            plt.show()

        return pred_label


# +--------------------------------------------------------------------------+
# |             UTILITIES                                                    |
# +--------------------------------------------------------------------------+


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


# +--------------------------------------------------------------------------+


def listImages(dir_path):
    """
    listImages
    ---
    List all the images (PNG, JPG, JPEG) in the given folder.

    It returns a list with the path of all images.
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"{dir_path} is not a directory!")

    img_path_list = []
    for im in os.listdir(dir_path):
        if im.endswith(".png") or im.endswith(".jpg") or im.endswith(".jpeg"):
            img_path_list.append(os.path.join(dir_path, im))

    return img_path_list


def main():
    """
    Perform inference using the trained model.

    Depending on the command line arguments passed to this script, the
    behavior will change.

    - No argument: the model will try to perform inference on all the images
    (PNG, JPEG, JPG) present in the current folder (from which the program is ran)
    - argv[1] == "rand": inference will be performed on a random image in the 'test' folder
    - argv[1] == "filename.png": inference will be performed on the specified image
    """

    script_folder = os.path.dirname(__file__)
    test_folder = os.path.join(script_folder, "test")
    disp_plot = False

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == "rand":
            # Random image among test ones to be tested if no specific path is passed
            disp_plot = True
            class_dir = os.path.join(
                test_folder, random.choice(os.listdir(test_folder))
            )
            test_img_path_list = [
                os.path.join(class_dir, random.choice(os.listdir(class_dir)))
            ]
        else:
            # Specify path of test image as command line argument
            disp_plot = True
            test_img_path_list = [str(sys.argv[1])]
    else:
        # Classify all (png) images in script folder
        test_img_path_list = listImages(".")

    if disp_plot:
        output_img_path = os.path.join(script_folder, "img", "inference.png")
    else:
        # No plot displayed if performing inference on many images!
        output_img_path = None

    model_path = os.path.join(script_folder, "0602-660603047-MACARIO.ZZZ")

    my_nn = MyNet(IMG_SHAPE, CLASS_MAP)

    if torch.backends.mps.is_available() and MPS:
        print("Using MPS")
        avail_device = torch.device("mps")
        my_nn.to(avail_device)
        # Load the saved model state dictionary
        my_nn.load_parameters(model_path, _device=avail_device)
    elif torch.cuda.is_available() and CUDA:
        print("Using CUDA!")
        avail_device = torch.device("cuda")
        my_nn.to(avail_device)
        # Load the saved model state dictionary
        my_nn.load_parameters(model_path, _device=avail_device)
    else:
        avail_device = None
        my_nn.load_parameters(model_path, _device=avail_device)

    # Iterate over images
    for im_pth in test_img_path_list:
        pred_class = my_nn.inference(
            im_pth, plot=disp_plot, img_path=output_img_path, _device=avail_device
        )
        # Display result of inference on stdout
        print(f"{os.path.basename(im_pth)}: {pred_class}")


if __name__ == "__main__":
    main()
