import numpy as np
import matplotlib.pyplot as plt
import sys


def step_function(x):
    return (np.heaviside(x, 1))


class Perceptron():
    def __init__(self, weights: np.ndarray, activation="step"):
        """
        Perceptron
        ---
        This class is used to model individual perceptrons in a neural network.

        ### Input parameters

        - weights: the array of weights. Its length is equal to the number of 
        inputs + 1, since the element in position 0 is the bias.
        - activation: string indicating the type of activation function (default 
        'sgn')
        """
        self.n_inputs = len(weights) - 1
        self.weights = weights

        if activation == "step":
            self.activation = step_function
        elif activation == "sgn":
            self.activation = np.sign
        else:
            raise ValueError(
                f"'{activation}' is not a valid activation function")

    def eval_output(self, inputs: np.array):
        """
        eval_output
        ---
        Evaluate the output of the neuron given its inputs.
        """
        if len(inputs) != len(self.weights) and len(inputs) != len(self.weights) - 1:
            raise ValueError(f"The input has wrong size!")
        elif len(inputs) == len(self.weights):
            assert inputs[
                0] == 1, "The first element of the inputs array should be 1 (it is x_0, the bias)"
            val = np.dot(inputs, self.weights)
        else:
            val = np.dot(np.insert(inputs, 0, 1), self.weights)
        return self.activation(val)


class NeuralNetwork():
    def __init__(self, neurons_per_layer: list, weights: list, activation="step"):
        """
        NeuralNetwork
        ---
        Fully connected neural network class, based on Perceptron objects.

        ### Input parameters

        - neurons_per_layer: 1D array containing for each layer the number of neurons
        - weights: 2D list, containing on each row the weights associated with each 
        neuron (including bias in the first column); the neurons should be ordered and
        the number of weights per neuron needs to be compliant with the values inside
        'neurons_per_layer'
        """
        self.n_inputs = len(weights[0]) - 1
        self.n_outputs = neurons_per_layer[-1]
        self.n_layers = len(neurons_per_layer)
        self.activation = activation

        assert len(weights) == sum(
            neurons_per_layer), f"{len(weights)} sets of weights were provided, but the total number of neurons is {sum(neurons_per_layer)}"

        # TODO: add check on the number of weight for neurons in every layer, according to
        # the n. of neurons in the layer before

        # Create neural network
        # 1st dim in the list is the layer, containing a sub-list of Perceptron objects
        # (top-down order)
        self.neurons = []
        ind_weights = 0
        for i in range(len(neurons_per_layer)):
            self.neurons.append([])
            for j in range(neurons_per_layer[i]):
                # Do the thing
                self.neurons[i].append(Perceptron(
                    weights=weights[ind_weights], activation=activation))
                ind_weights += 1

    def eval_output(self, inputs: np.ndarray):
        """
        eval_output
        ---
        Evaluate the output of the neural network given the provided input.
        """

        assert len(
            inputs) == self.n_inputs, f"The input size should be {self.n_inputs}, provided input is {inputs}"

        for i in range(len(self.neurons)):
            if i == 0:
                arr_in = np.array(inputs)
            else:
                arr_in = np.array(lst_outputs)
            lst_outputs = []
            for j in range(len(self.neurons[i])):
                # For each neuron at the layer, evaluate the output of the
                lst_outputs.append(self.neurons[i][j].eval_output(arr_in))

        return np.array(lst_outputs)


if __name__ == "__main__":
    my_weights = [[1, 1, -1], [1, -1, -1], [0, -1, 0], [-1.5, 1, 1, -1]]

    my_nn = NeuralNetwork(neurons_per_layer=[3, 1], weights=my_weights)

    # print(f"N. inputs: {my_nn.n_inputs}")
    # pt = np.array([1, 1])
    # print(my_nn.eval_output(pt))

    np.random.seed(660603047)

    # Generate 1000 points in [-2, 2]^2
    points = np.random.rand(1000, 2) * 4 - 2
    outputs = []
    for i in range(1000):
        outputs.append(my_nn.eval_output(points[i, :])[0])

    arr_outputs = np.array(outputs)

    # Plot:
    red_pt = points[arr_outputs == 1, :]
    blue_pt = points[arr_outputs == 0, :]

    plt.figure()
    plt.plot(red_pt[:, 0], red_pt[:, 1], 'ro', label="output = 1")
    plt.plot(blue_pt[:, 0], blue_pt[:, 1], 'bo', label="output = 0")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural network output")
    if len(sys.argv) > 1:
        plt.savefig(sys.argv[1])
    plt.show()
