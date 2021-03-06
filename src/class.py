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
    def __init__(self, input_size, hidden_size, num_classes, p=0):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, num_classes, bias=False)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.dropout(self.fc2(out)))
        out = F.relu(self.dropout(self.fc3(out)))
        out = self.fc4(out)
        return out