#!/usr/bin/env python3
# https://github.com/eugeniaring/Medium-Articles/blob/main/Pytorch/denAE.ipynb

import os  # this module will be used just to create directories in the local filesystem
import random  # this module will be used to select random samples from a collection
from typing import List

import matplotlib.pyplot as plt
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd  # this module is useful to work with tabular data
import plotly.express as px
import plotly.io as pio
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm  # this module is useful to plot progress bars

MPS = True
CUDA = True
EPOCHS = 30  # FIXME: re-set to 30

VERB = True
PLOTS = False


script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "dataset")
### With these commands the train and test datasets, respectively, are downloaded
### automatically and stored in the local "data_dir" directory.
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)


fig, axs = plt.subplots(5, 5, figsize=(8, 8))
for ax in axs.flatten():
    # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
    img, label = random.choice(train_dataset)
    ax.imshow(np.array(img), cmap="gist_gray")
    ax.set_title("Label: %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
if PLOTS:
    plt.show()

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Set the train transform
train_dataset.transform = train_transform
# Set the test transform
test_dataset.transform = test_transform

m = len(train_dataset)

# random_split randomly split a dataset into non-overlapping new datasets of given lengths
# train (55,000 images), val split (5,000 images)
train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])

print(f"Data sets:")
print(f" - Training set: {len(train_data)} elements")
print(f" - Validation set: {len(val_data)} elements")
print(f" - Test set: {len(test_dataset)} elements")

batch_size = 256

# The dataloaders handle shuffling, batching, etc...
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        ### Flatten layer
        # self.flatten = torch.flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim),
        )

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x


### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 4

encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr = 0.001  # Learning rate


params_to_optimize = [
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
]

if torch.backends.mps.is_available() and MPS:
    print("Using MPS")
    device = torch.device("mps")
elif torch.cuda.is_available() and CUDA:
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
# print(f'Selected device: {device}')

optim = torch.optim.Adam(params_to_optimize, lr=lr)

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)
# model.to(device)


def add_noise(inputs, noise_factor=0.3):
    noise = inputs + torch.randn_like(inputs) * noise_factor
    noise = torch.clamp(noise, 0.0, 1.0)
    return noise


### Training function
def train_epoch_den(
    encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3
):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    # with "_" we just ignore the labels (the second element of the dataloader tuple)
    for image_batch, _ in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        image_noisy = add_noise(image_batch, noise_factor)
        image_noisy = image_noisy.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs_den(
    encoder,
    decoder,
    n=5,
    noise_factor=0.3,
    savefig=False,
    disp=False,
    epoch_num=None,
):
    img_folder = os.path.join(os.path.dirname(__file__), "img")
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        img = test_dataset[i][0].unsqueeze(0)
        image_noisy = add_noise(img, noise_factor)
        image_noisy = image_noisy.to(device)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            rec_img = decoder(encoder(image_noisy))

        plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original images")
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Corrupted images")

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed images")
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.3
    )
    plt.tight_layout()
    if savefig:
        if epoch_num is None:
            plt.savefig(os.path.join(img_folder, "autoenc_outputs.png"))
        else:
            plt.savefig(os.path.join(img_folder, f"autoenc_outputs_ep{epoch_num}.png"))
    if PLOTS:
        plt.show()
    plt.close()


### Training cycle
noise_factor = 0.3
num_epochs = EPOCHS
history_da = {"train_loss": [], "val_loss": []}

for epoch in range(num_epochs):
    print("EPOCH %d/%d" % (epoch + 1, num_epochs))
    ### Training (use the training function)
    train_loss = train_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optim,
        noise_factor=noise_factor,
    )
    ### Validation  (use the testing function)
    val_loss = test_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=valid_loader,
        loss_fn=loss_fn,
        noise_factor=noise_factor,
    )
    # Print Validationloss
    history_da["train_loss"].append(train_loss)
    history_da["val_loss"].append(val_loss)
    print(
        "\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}".format(
            epoch + 1, num_epochs, train_loss, val_loss
        )
    )
    plot_ae_outputs_den(
        encoder,
        decoder,
        noise_factor=noise_factor,
        savefig=True,
        disp=False,
        epoch_num=epoch,
    )


# +--------------------------------------------------------------------------+
# Put your image generator here
# +--------------------------------------------------------------------------+


def generateRandomImages(
    n_img: int,
    latent_space_dim: int,
    decoder: Decoder,
    device: torch.device,
) -> torch.Tensor:
    """
    generateRandomImages
    ---
    Generate `n_img` random images through the decoder by feeding it
    random vectors in the latent space.

    ### Input parameters
    - n_img: number of images to be generated.
    - latent_space_dim: dimension of the latent space (encoded)
    - decoder: decoding neural net.

    ### Output parameter
    - torch.tensor containing the n_img produced outputs (shape:
    [n_img, 1, 28, 28])
    """
    tensor_shape = (n_img, latent_space_dim)
    rand_gauss_tensor = torch.randn(tensor_shape)
    # Move to device
    rand_gauss_tensor = rand_gauss_tensor.to(device)

    decoder.eval()
    dev_decoder = decoder.to(device)
    # Feed the random vectors to the decoder to get images
    outs = dev_decoder(rand_gauss_tensor)

    return outs


# +--------------------------------------------------------------------------+

gener_images = generateRandomImages(9, d, decoder, device)
# Move generated tensor to CPU
gener_img_cpu = gener_images.detach().cpu().clone().numpy()

# Plot them in a 3x3 matrix-shaped plot
fig, axs = plt.subplots(3, 3, figsize=(8, 8))

ind = 0
for ax in axs.flatten():
    ax.imshow(
        np.array(gener_img_cpu[ind].reshape(28, 28)),
        cmap="gist_gray",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ind += 1
fig.suptitle("Random images generated from gaussian noise", fontsize=20)
plt.tight_layout()
img_folder = os.path.join(os.path.dirname(__file__), "img")
plt.savefig(os.path.join(img_folder, "generated_from_decoder.png"))
if PLOTS:
    plt.show()

# +--------------------------------------------------------------------------+
# Put your clustering accuracy calculation here
# +--------------------------------------------------------------------------+


def loadingBar(
    current_iter: int,
    tot_iter: int,
    n_chars: int = 10,
    ch: str = "=",
    n_ch: str = " ",
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


# First, obtain all encoder outputs for all elements of the training set
# dl_train_set = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
dl_train_set = train_loader
encoded_images = []
labels_batches = []
it = 0
n_train = len(train_data)
encoder.eval()
print("\nExtracting compressed representation of training set elements:")
with torch.no_grad():
    for images, labels in dl_train_set:
        img = images.to(device)

        out = encoder(img.unsqueeze(0))

        labels_batches.append(labels.item())
        encoded_images.append(out.cpu().numpy())

        print(loadingBar(it, n_train, 20), f" {it}/{n_train}", end="\r")

        it += 1

encoded_train = np.concatenate(encoded_images)
labels_train = np.concatenate(labels_batches)

labels_train_arr = np.array(labels_train)
encoded_train_arr = np.array(encoded_train)

# Apply K-Means clustering on `encoded_train` - the class of the cluster is
# chosen by majority
n_clusters = 10
print("Start clustering                            ")
clt = KMeans(n_clusters, n_init=10, max_iter=1000, tol=1e-6)
clusters_train = clt.fit_predict(encoded_train_arr)
print("Finish clustering")

print(labels_train_arr)
print(clusters_train)


# TODO: map labels to clusters (majority polling)
def mapClusters(clusters: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Map the cluster labels to the actual class labels of the items.

    The returned array contains in element 'i' the cluster label associated
    with dataset label 'i'.
    """
    clust_labels, counts = np.unique(clusters, return_counts=True)
    class_labels = np.unique(labels)

    # Ensure the same amount of classes is present
    assert len(clust_labels) == len(class_labels)

    # Sort the class labels in descending order according to the number of
    # elements of that cluster.
    # This way, the clusters with more elements will 'choose' the label first,
    # contributing to a higher accuracy.
    clust_labels = clust_labels[np.argsort(-1 * counts)]

    mapping = -1 * np.ones((len(class_labels),), dtype=int)
    for c in clust_labels:
        print(f"Cluster {c} ", end="")
        assoc_labels = labels[clusters == c]
        labels_counts = np.bincount(assoc_labels)
        # If not all classes appear in the current cluster, labels_count has
        # less than 10 elements (this will break while cycle)
        labels_counts_full = np.zeros((len(class_labels),))
        labels_counts_full[: len(labels_counts)] = labels_counts
        sort_freq = np.argsort(-1 * labels_counts_full)

        # Idea: to prevent assigning 2 clusters the same label, ensure that
        # label is assigned only if not already taken
        ind = 0
        while ind < len(sort_freq) and mapping[sort_freq[ind]] != -1:
            ind += 1
            print("•", end="")
        print()
        mapping[sort_freq[ind]] = c

    assert all(mapping != -1)

    if VERB:
        print("Mappings:")
        print(mapping)

    return mapping


# In cluster_map, element 'i' corresponds the cluster label associated with
# class 'i' (i.e., the digit 'i')
cluster_map = mapClusters(clusters_train, labels_train_arr)
print("Mappings: {}".format(cluster_map))

# Evaluate accuracy
n_exact = 0
for i in range(len(labels_train_arr)):
    if cluster_map[labels_train_arr[i]] == clusters_train[i]:
        n_exact += 1

acc_cluster = n_exact / labels_train_arr.shape[0]
print(f"Clustering accuracy: {acc_cluster}")

# Get centroids of the clusters and generate image of number '5'
centroids = clt.cluster_centers_
centr_5 = centroids[cluster_map[5]]
centr_5 = centr_5.reshape((1, len(centr_5)))

if VERB:
    print(centr_5)

# Make it a tensor
c5_tens = torch.from_numpy(centr_5)
c5_tens = c5_tens.to(device)
decoder.eval()
out_tens = decoder(c5_tens)
out_img = out_tens.detach().cpu().clone().numpy()
out_img = np.squeeze(out_img)

plt.figure()
plt.imshow(out_img, cmap="gist_gray")
plt.title("Image generated from centroid of cluster '5'")
plt.tight_layout()
plt.savefig(os.path.join(img_folder, "output_n_5.png"))
plt.show()
