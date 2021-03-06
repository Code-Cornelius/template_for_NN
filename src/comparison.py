import time
from priv_lib_util.tools import benchmarking

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

def analyze_neural_network(data_X, data_Y,
                           indices_train_X, indices_train_Y,
                           indices_test_X, indices_test_Y,
                           parameters_for_training,
                           silent=False,
                           plot = False):
    """

    Args:
        data_X:
        data_Y:
        indices_train_X:
        indices_train_Y:
        indices_test_X:
        indices_test_Y:
        parameters_for_training:
        silent:
        plot:

    Returns:

    """


                           # data_train_X, data_train_Y,
                           # data_test_X, data_test_Y,
                           # batch_size=BATCH_SIZE,
                           # learning_rate=LEARNING_RATE,
                           # epochs=N_EPOCHS,
                           # silent=False):
    # trains (training + validation)
    # test with a confirmation set.

    # plot the comparison and the training.

    start_training = time.time()


    net, a, b, ta, tb = nn_kfold_train(data_train_X, data_train_Y, data_test_X, data_test_Y, k, learning_rate,
                                       batch_size, epochs=epochs, silent=silent)

    # plot the loss and accuracy evolution through the epochs
    nn_plot(a, ta, b, tb)

    # compute the last accuracy
    data_train_Y_pred = nn_predict(net, data_train_X)
    data_test_Y_pred = nn_predict(net, data_test_X)

    # results of the performance
    result_function("Neural Network", data_train_Y, data_train_Y_pred, data_test_Y, data_test_Y_pred)

    # print elapsed time
    benchmarking.time_print_elapsed_time(start_training, time.time(), title="Time for training the NN: ")

    return net
