# for neural networks
import torch
import torch.nn as nn


# the class of NN
class NeuralNet(nn.Module):

    def __init__(self, list_hidden_sizes, hidden_sizes, output_size, list_biases, activation_functions, dropout=0):
        """
        Constructor for Neural Network
        Args:
            list_hidden_sizes: the size of the input layer
            hidden_sizes: the input sizes for each hidden layer + output of last hidden layer
            output_size: the output size of the neural network
            list_biases: list of booleans for specifying which layers use biases
            activation_functions: list of activation functions for each layer
            dropout: dropout rate
        """
        super(NeuralNet, self).__init__()
        # check the inputs
        assert len(list_biases) == len(hidden_sizes) + 1
        self.input_size = list_hidden_sizes
        self.list_hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.list_biases = list_biases
        self.activation_functions = activation_functions

        # array of fully connected layers
        self.fcs = nn.ModuleList()

        # initialise the input layer
        self.fcs.append(nn.Linear(self.input_size, self.list_hidden_sizes[0], self.list_biases[0]))#.cuda())

        # initialise the hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.fcs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], self.list_biases[i + 1]))#.cuda())

        # initialise the output layer
        self.fcs.append(nn.Linear(self.list_hidden_sizes[-1], self.output_size, self.list_biases[-1]))#.cuda())

        # initialise dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """method that passes through the layers."""
        # pass through the input layer
        out = self.activation_functions[0](self.fcs[0](x))

        # pass through the hidden layers
        for layer_index in range(1, len(self.list_hidden_sizes) - 1):
            out = self.activation_functions[layer_index](self.dropout(self.fcs[layer_index](out)))

        # pass through the output layer
        out = self.fcs[-1](out)
        return out

    def prediction(self, out):
        """returns the class predicted for each element of the tensor."""
        # gets the class that is max probability
        return torch.max(out,1)[1]
