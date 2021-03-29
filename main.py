#todo requirement
# make the workers work, in particular check if they work in Linux.
# clean the plots. It would be good to have all the plots on the same graph,
# and save the model and use it to check the accuracy total.
# the parameters need to put in NNTrainParameters dict with parameters for optimiser.


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

from src.Neural_Network.NN_plots import *
from src.Neural_Network.NN_fct import *
from src.Neural_Network.NNTrainParameters import *
from src.Neural_Network.Fully_connected_NN import *

# set seed for pytorch.
torch.manual_seed(42)
np.random.seed(42)


from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
train_Y = pd.DataFrame(train_y)

test_X = pd.DataFrame(test_X.reshape(10000, 28 * 28))
test_Y = pd.DataFrame(test_y)

input_size = 28 * 28
hidden_sizes = [1000, 1000]
output_size = 10
biases = [True, True, True]
activation_functions = [F.relu, F.relu]
dropout = 0.4
epochs = 10
batch_size = 2000
# WIP
optimiser = torch.optim.SGD
criterion = nn.CrossEntropyLoss(), # criterion = nn.CrossEntropyLoss() # criterion = nn.NLLLoss()


pytorch_device_setting()
dict_optimiser = {"lr": 0.001, "weight_decay" : True}
parameters_for_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                            criterion=criterion , optimiser=optimiser,
                                            dict_params_optimiser = dict_optimiser)

train_X = torch.from_numpy(train_X.values).float()
train_Y = torch.from_numpy(train_Y.values).long()
test_X = torch.from_numpy(test_X.values).float()
test_Y = torch.from_numpy(test_Y.values).long()

if __name__ == '__main__':
    net, mean_training_accuracy, mean_validation_accuracy, mean_training_losses, mean_validation_losses = \
        nn_kfold_train(train_X, train_Y, input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
                       parameters_for_training=parameters_for_training, early_stopper_validation=None, nb_split=3,
                       shuffle=True, silent=False)

    nn_plot(mean_training_accuracy[0,:], mean_training_losses[0,:], mean_valid_acc=mean_validation_accuracy[0,:], mean_valid_losses=mean_validation_losses[0,:])
    confusion_matrix_creator(train_Y, nn_predict(net, train_X), range(10))
    confusion_matrix_creator(test_Y, nn_predict(net, test_X), range(10))

    plt.show()

# analyze_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5, epochs=30, silent=True)
#
# # Activation Function
# batch_size = 128
# learning_rate = 0.005
# epochs = 30
# hidden_size = 16
# num_layers = 2
# dropout = 0
# norm = False
# activ_function = "tanh"
# version = 0
# optim = "sgd"
#
# analyze_convolution_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5,
#                                    batch_size, learning_rate, epochs,
#                                    hidden_size, num_layers, dropout, norm, activ_function, version, optim, True)
