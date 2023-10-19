import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, labels, classes):
    """
    show_images
    ---
    Display some random training images with the associated label.
    """
    plt.figure((10, 7))
    for i in range(len()):
        pass
    plt.show()


class Net(nn.Module):
    def __init__(self):
        # This class inherits from nn.Module
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, 5
        )  # 3x input channels, 16 output, kernel size = 5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 ouptut classes (CIFAR-10)

    def forward(self, x):
        """
        forward
        ---
        Get NN output for input x.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)  # Flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == "__main__":
    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose()
    
    # Load CIFAR-10 data set (train & test)

    # [...]

    n_epochs = 10
    for epoch in range(n_epochs):
        running_loss = 0.
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.stop()

            running_loss += loss.item()

        train_acc = 
