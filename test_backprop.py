import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEBUG = True
VERB = True


class MyNet(nn.Module):
    def __init__(self, input_shape: int, act_func: Callable = F.relu):
        super(MyNet, self).__init__()

        self.input_size = input_shape
        self.out_dim = 1

        # Activation function
        self.act_func = act_func

        # Layer Definition
        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, self.out_dim)

        # Variables for displaying performance
        self.n_epochs = None
        self.loss_train = None
        self.loss_test = None
        self.acc_train = None
        self.acc_test = None

    def forward(self, x_in: torch.Tensor, dropout: bool = True) -> torch.Tensor:
        """
        Forward propagation in the neural network.

        Args
            x: input of the network
            dropout: flag to select dropout

        Returns
            y: output of the network [array]
        """
        x = self.act_func(self.fc1(x_in))
        x = self.act_func(self.fc2(x))
        y = self.fc3(x)
        return y


class myData(Dataset):
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0]
        self.train_inputs = inputs
        self.train_outputs = outputs

    def __len__(self):
        return self.train_inputs.shape[0]

    def __getitem__(self, idx):
        return self.train_inputs[idx], self.train_outputs[idx]


def train(model, n_epochs, train_dl, device):
    """Train the model"""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1, weight_decay=0.01)
    obj_func = nn.MSELoss()

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            outputs = model(inputs)
            loss = obj_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

        print(f"> Loss: {loss.item():.4f}")

        # for name, param in model.named_parameters():
        #     print(f"Gradient for {name}: {param.grad}")
    print("")


def validate(model, valid_dl, device):
    """Performance metric: MSE"""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        num = 0
        total = 0
        for inputs, targets in valid_dl:
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            outputs = model(inputs)
            total += targets.size(0)
            num += torch.sum((outputs - targets) ** 2)

    mse = num / total
    print(f"MSE of the model on the training data: {mse}")


def main(args):
    # Data points: function
    x_tr = torch.rand(args.n_train, 1)
    noise_tr = torch.randn(args.n_train, 1) / 100
    y_tr = x_tr**2 + noise_tr
    train_dataset = myData(x_tr, y_tr)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    x_val = torch.rand(args.n_val, 1)
    noise_val = torch.randn(args.n_val, 1) / 100
    y_val = x_val**2 + noise_val
    val_dataset = myData(x_val, y_val)
    val_dl = DataLoader(val_dataset, shuffle=True)

    model = MyNet(input_shape=1).to(args.device)
    train(model, n_epochs=args.n_epochs, train_dl=train_dl, device=args.device)

    validate(model, valid_dl=val_dl, device=args.device)

    if args.plots:
        model.eval()
        x = x_val.numpy()
        y = y_val.numpy()
        with torch.no_grad():
            x_val_dev = x_val.to(args.device)
            y_mod = model(x_val_dev).cpu().numpy()

        plt.figure()
        plt.plot(x, y, "r.", label="Actual")
        plt.plot(x, y_mod, "b.", label="Approx")
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Comparison")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="if set, produce plot for validation data"
    )
    parser.add_argument(
        "-v", "--verb", action="store_true"
    )
    parser.add_argument(
        "--n-train", type=int, default=10000, help="number of training items"
    )
    parser.add_argument(
        "--n-val", type=int, default=1000, help="number of validation items"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="training batch size"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device on which to load the model"
    )
    args = parser.parse_args()
    main(args)
