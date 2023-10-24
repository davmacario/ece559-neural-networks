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
from typing import Callable

VERB = False
PLOTS = False
MPS = True
CUDA = True


class MyNet(nn.Module):
    """
    MyNet
    ---
    Neural network used for shape classification.
    """

    def __init__(self, input_shape: (int, int), n_classes: int):
        """
        Initialize the neural network.

        ### Input parameters
        - input_shape:
        - n_classes:

        ### Network structure (layers)
        -
        """
        super(MyNet, self).__init__()

        self.input_size = input_shape
        self.n_classes = n_classes
        # Define layers
        self.pool_halve = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 3)

        self.len_1st_fc = int(50 * 10 * 10)
        self.fc1 = nn.Linear(self.len_1st_fc, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc3 = nn.Linear(80, n_classes)

    def forward(
        self, x, act_func: Callable = nn.functional.relu, softmax_out: bool = False
    ):
        """
        forward
        ---
        Forward propagation in the neural network

        ## Input parameters
        - x: input of the network
        - act_func: activation function to be used in the intermediate layers
        - softmax_out: flag to indicate whether to apply softmax function at the
        output of the network

        ### Output parameters
        - y: output of the network [array]
        """
        y = self.pool_4(x)
        y = self.pool_halve(act_func(self.conv1(y)))
        y = self.pool_halve(act_func(self.conv2(y)))
        if VERB:
            print(y.shape)
        y = y.view(-1, self.len_1st_fc)
        y = act_func(self.fc1(y))
        y = act_func(self.fc2(y))
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
        for epoch in range(n_epochs):  # Change the number of epochs as needed
            running_loss = 0.0
            # print(loadingBar(epoch, n_epochs, 20), end="\r")
            for i, data in enumerate(train_dataloader, 0):
                print(
                    f"> Current epoch - {loadingBar(i, len(train_dataloader), 30)} {round(100 * i / len(train_dataloader), 3)}%",
                    end="\r",
                )

                inputs, labels = data

                if _device is not None:
                    inputs = inputs.to(_device)
                    labels = labels.to(_device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = obj_func(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(
                f"Epoch {epoch + 1} done!                                                        ",
                end="\r",
            )

            train_accuracy = eval_accuracy(self, train_dataloader, _device)
            if test_accuracy is not None:
                test_accuracy = eval_accuracy(self, test_dataloader, _device)
                print(
                    f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}, Train Accuracy: {train_accuracy}%"
                )

        print("Finished Training!")
        torch.save(self.state_dict(), model_path)
        print("Model stored at {}".format(model_path))


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
                os.path.join(ds_path, fname), os.path.join(tr_path, class_curr, fname)
            )
        else:
            # Place current image in test set
            shutil.copy(
                os.path.join(ds_path, fname), os.path.join(te_path, class_curr, fname)
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
    - train_path: path of the trainin set folder
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
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            if _device is not None:
                inputs = inputs.to(_device)
                labels = labels.to(_device)
            outputs = network(inputs)
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


def main():
    script_folder = os.path.dirname(__file__)
    dataset_path = os.path.join(script_folder, "output")
    train_path = os.path.join(script_folder, "train")
    test_path = os.path.join(script_folder, "test")
    images_folder = os.path.join(script_folder, "img")
    n_training = 8000

    try:
        splitDataset(n_training, dataset_path)
        print("Training and test sets generated!")
    except FileExistsError:
        print("Training and test set already split!")

    dl_train, classes_map = importDataset(train_path)
    dl_test, _ = importDataset(test_path)

    tr_img, tr_labels = next(iter(dl_train))

    if VERB:
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

    if VERB:
        print(tr_img.shape[2:4])

    my_nn = MyNet(tr_img.shape[2:4], len(classes_map.keys()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.001, momentum=0.9)

    model_path = os.path.join(script_folder, "0602-660603047-MACARIO.ZZZ")

    if torch.backends.mps.is_available() and MPS:
        print("Using MPS!")
        mps_device = torch.device("mps")
        my_nn.to(mps_device)
        my_nn.train_nn(
            dl_train, optimizer, criterion, 10, dl_test, model_path, mps_device
        )
    elif torch.cuda.is_available() and CUDA:
        print("Using CUDA!")
        cuda_device = torch.device("cuda")
        my_nn.to(cuda_device)
        my_nn.train_nn(
            dl_train, optimizer, criterion, 10, dl_test, model_path, cuda_device
        )
    else:
        my_nn.train_nn(dl_train, optimizer, criterion, 10, dl_test, model_path)


if __name__ == "__main__":
    main()
