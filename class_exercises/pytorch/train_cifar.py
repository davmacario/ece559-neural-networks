import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the neural network
net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Define data transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Define a function to calculate accuracy
def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Train the neural network
for epoch in range(10):  # Change the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_accuracy = calculate_accuracy(trainloader)
    test_accuracy = calculate_accuracy(testloader)
    print(
        f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%"
    )

print("Finished Training")
torch.save(net.state_dict(), "cifar10_model.pth")


# Function to show images with true and predicted labels
def show_images_with_labels(loader, num_images=5):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    fig, axs = plt.subplots(1, num_images, figsize=(12, 2))
    for i in range(num_images):
        image = images[i] / 2 + 0.5  # Unnormalize the image
        image = image.numpy().transpose((1, 2, 0))
        label = classes[labels[i]]
        pred_label = classes[predicted[i]]

        axs[i].imshow(image)
        axs[i].set_title(f"True: {label}\nPredicted: {pred_label}")
        axs[i].axis("off")
    plt.show()


# Show images with true and predicted labels
show_images_with_labels(testloader, num_images=5)
