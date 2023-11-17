# HW6 notes

## Part a - Code description

Explaination of the `hw6.py` script.

* Pre-processing
  * Import MNIST data set (train and test set)
    * Show some random training elements
  * Assign `transform` objects to the two sets (simply convert them to Tensors)
  * Split training data into train + val (48,000 & 12,000)
  * Set batch size = 256
  * Create dataloaders (<- maybe **move variables on device here**)
* Define encoder (NN)
  * Decide layers (conv + fc)
* Define decoder (NN):
* Initialize objects (128 x 128 inputs) - *latent space* has dimension 4
* Set loss function: MSE
* Select device
  * Move encoder and decoder to device
* Init. optimizer (Adam, w/ learning rate:`lr`)
* Def. `add_noise` function

## Explaining mapping algorithm

