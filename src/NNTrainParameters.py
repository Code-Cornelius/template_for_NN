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

