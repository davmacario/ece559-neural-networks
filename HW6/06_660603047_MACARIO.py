#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

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
# dl_train_set = train_loader


def compressDataSet(
    dl: DataLoader, encoder: Encoder, device: torch.device
) -> tuple[NDArray, NDArray]:
    """
    Compress a data set using the encoder.

    ### Input parameters
    - dl: dataloader containing the
    """
    encoded_images = []
    labels_batches = []
    it = 0
    n_train = len(dl)
    encoder.eval()
    print("\nExtracting compressed representation of training set elements:")
    with torch.no_grad():
        for images, labels in dl:
            img = images.to(device)

            out = encoder(img)

            labels_batches.append(labels)
            encoded_images.append(out.cpu().numpy())

            print(loadingBar(it, n_train, 20), f" {it}/{n_train}", end="\r")

            it += 1

    encoded = np.concatenate(encoded_images)
    labels = np.concatenate(labels_batches)

    labels_arr = np.array(labels)
    encoded_arr = np.array(encoded)

    return encoded_arr, labels_arr


encoded_train_arr, labels_train_arr = compressDataSet(train_loader, encoder, device)

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
