import random
import numpy as np
import matplotlib.pyplot as plt

"""
Perceptron Training Algorithm
"""


def step_function(x: float):
    return np.heaviside(x, 1)


def count_misclassifications(w, x, d):
    """
    count_misclassifications
    ---
    Count how many data set elements are not correctly classified 
    by the current weights (w).
    """
    assert len(
        d) == x.shape[1], f"The dimensions of x and d do not match ({len(d)} vs. {x.dim(1)})"
    d_prime = np.array([int(np.dot(x[:, i], w) >= 0)
                       for i in range(x.shape[1])])
    return np.sum(np.absolute(d - d_prime))


def perceptron_training_algorithm(eta: float, w_start_arr: np.ndarray, train_set: dict):
    epoch = 0
    all_class_corr = False

    x = train_set["X"]
    class_labels = train_set["d"]
    w_curr = w_start_arr
    n_misclassifications = count_misclassifications(w_curr, x, class_labels)
    print(f"Initial weights: [{w_curr[0]}, {w_curr[1]}, {w_curr[2]}]")
    print(f"N. misclassifications: {n_misclassifications}")

    while not all_class_corr:

        n_misclassifications = 0

        for i in range(x.shape[1]):
            # Evaluate the output with x_i as input
            d_prime = step_function(np.dot(x[:, i], w_curr))

            if d_prime == 0 and class_labels[i] == 1:
                n_misclassifications += 1
                w_curr = w_curr + eta * x[:, i]
            elif d_prime == 1 and class_labels[i] == 0:
                n_misclassifications += 1
                w_curr = w_curr - eta * x[:, i]

            # print(
            #     f"Weights at epoch {epoch}:\nw_0 = {w_curr[0]}\nw_1 = {w_curr[1]}\nw_2 = {w_curr[2]}")

        epoch += 1
        all_class_corr = (n_misclassifications == 0)

    print(f"Training finished! Number of epochs: {epoch}")
    return w_curr


if __name__ == "__main__":
    random.seed(660603047)
    np.random.seed(660603047)

    N_x = 1000  # Number of training set elements (range of 'i')

    # Pick w_0:
    w_0 = random.uniform(-0.25, 0.25)
    # Pick w_1:
    w_1 = random.uniform(-1, 1)
    # Pick w_2:
    w_2 = random.uniform(-1, 1)
    W = np.array([w_0, w_1, w_2])

    # Pick 100 (column) vectors 'x'
    X = np.random.uniform(-1, 1, (2, N_x))
    X_with_ones = np.vstack((np.ones((1, N_x)), X))

    # Decide whether each vector is in S1 or S0:
    # d = 1 -> in S1, d = 0 -> in S0
    d = np.array([int(np.dot(X_with_ones[:, i], W) >= 0)
                  for i in range(N_x)])

    # Plot decision region for x1 and x2
    x1_plt = np.linspace(-1, 1, 10000)
    x2_plt = (-w_0 - w_1 * x1_plt) / w_2

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(x1_plt, x2_plt, 'g', label='Decision boundary')
    ax.plot(X[0, d == 1], X[1, d == 1],
            'or', label=r'Points in $S_{0}$')
    ax.plot(X[0, d == 0], X[1, d == 0],
            'ob', label=r'Points in $S_{1}$')
    ax.grid()
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.title("Input training points")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    plt.show()

    # Perceptron Training Algorithm
    w_prime = np.random.uniform(-1, 1, (3,))
    train_set = {
        "X": X_with_ones,
        "d": d
    }
    est_w = perceptron_training_algorithm(
        eta=1, w_start_arr=w_prime, train_set=train_set)

    print(f"Actual vs. estimated weights:")
    print(f"w_0 = {w_0} vs. {est_w[0]}")
    print(f"w_0 = {w_1} vs. {est_w[1]}")
    print(f"w_0 = {w_2} vs. {est_w[2]}")

    # Plot updated decision region
    x2_prime_plt = (-est_w[0] - est_w[1] * x1_plt) / est_w[2]
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(x1_plt, x2_plt, 'g', label='Decision boundary')
    ax.plot(X[0, d == 1], X[1, d == 1],
            'or', label=r'Points in $S_{0}$')
    ax.plot(X[0, d == 0], X[1, d == 0],
            'ob', label=r'Points in $S_{1}$')
    ax.plot(x1_plt, x2_prime_plt, color='black',
            linestyle='dashed', label="Estimated boundary")
    ax.grid()
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.title("Input training points")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    plt.show()
