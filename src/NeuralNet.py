# for neural networks
import torch.nn as nn


# the class of NN
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, biases, activation_functions, p=0):
        """
        Constructor for Neural Network
        Args:
            input_size: the size of the input layer
            hidden_sizes: the input sizes for each hidden layer + output of last hidden layer
            output_size: the output size of the neural network
            biases: list of booleans for specifying which layers use biases
            activation_functions: list of activation functions for each layer
            p: dropout rate
        """
        super(NeuralNet, self).__init__()
        # check the inputs
        assert len(biases) == len(hidden_sizes) + 1
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.biases = biases
        self.activation_functions = activation_functions

        # array of fully connected layers
        self.fcs = []

        # initialise the input layer
        self.fcs.append(nn.Linear(self.input_size, self.hidden_sizes[0], self.biases[0]))

        # initialise the hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.fcs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], self.biases[i + 1]))

        # initialise the output layer
        self.fc4 = nn.Linear(self.hidden_sizes[-1], self.output_size, self.biases[-1])

        # initialise dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        # pass through the input layer
        out = self.activation_functions[0](self.fcs[0](x))

        # pass through the hidden layers
        for layer_index in range(1, len(self.hidden_sizes) - 1):
            out = self.activation_functions[layer_index](self.dropout(self.fcs[layer_index](out)))

        # pass through the output layer
        out = self.fcs[-1](out)
        return out
