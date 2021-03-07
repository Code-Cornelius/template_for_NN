# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import seaborn as sns  # for the display
import pandas as pd  # for dataframes
import time  # computational time
import scipy.stats as si

# from useful_functions import *

# for neural networks
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import sklearn.model_selection
import torchvision
from torchvision import transforms

from src.plots import *
from src.nn_fct import  *
from src.NNTrainParameters import *
from src.NeuralNet import *

from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
train_Y = pd.DataFrame(train_y)

test_X = pd.DataFrame(test_X.reshape(10000, 28 * 28))
test_Y = pd.DataFrame(test_y)




analyze_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5, epochs=30, silent=True)

# Activation Function
batch_size = 128
learning_rate = 0.005
epochs = 30
hidden_size = 16
num_layers = 2
dropout = 0
norm = False
activ_function = "tanh"
version = 0
optim = "sgd"

analyze_convolution_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5,
                                   batch_size, learning_rate, epochs,
                                   hidden_size, num_layers, dropout, norm, activ_function, version, optim, True)