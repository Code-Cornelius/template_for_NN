import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from sklearn.model_selection import GridSearchCV
import time
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.metrics

# for clustering
import sklearn.cluster

# for graphs:
import networkx as nx
import csv
from operator import itemgetter

# some statistics
import statistics
import seaborn as sns

sns.set()

# for neural networks
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


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


class NNTrainParameters:

    def __init__(self, batch_size, learning_rate, epochs, criterion, optimiser):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = criterion
        self.optimiser = optimiser

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        if isinstance(new_batch_size, int) and new_batch_size >= 0:
                self._batch_size = new_batch_size
        else:
            raise TypeError(f'Argument is not an {str()}.')

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        if isinstance(new_learning_rate, float):
                self._learning_rate = new_learning_rate
        else:
            raise TypeError(f'Argument is not an {str()}.')

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, new_epochs):
        if isinstance(new_epochs, int) and new_epochs >= 0:
                self._epochs = new_epochs
        else:
            raise TypeError(f'Argument is not an {str()}.')

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, new_criterion):
        self._criterion = new_criterion

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, new_optimiser):
        self._optimiser = new_optimiser
