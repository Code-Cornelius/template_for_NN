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


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    References:
            https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return False

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
