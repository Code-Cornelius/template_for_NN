# todo requirement
#  make the workers work, in particular check if they work in Linux.
#  and save the model and use it to check the accuracy total.
from priv_lib_plot import APlot

import test_nn_kfold_train

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

from src.Neural_Network.Fully_connected_NN import factory_parametrised_FC_NN
from src.Neural_Network.NNTrainParameters import NNTrainParameters
from src.Neural_Network.NN_fcts import pytorch_device_setting, nn_predict
from src.Neural_Network.NN_kfold_training import nn_kfold_train
from src.Neural_Network.NN_plots import confusion_matrix_creator, nn_plot_train_loss_acc
from src.Training_stopper.Early_stopper_training import Early_stopper_training
from src.Training_stopper.Early_stopper_validation import Early_stopper_validation


# set seed for pytorch.
torch.manual_seed(42)
np.random.seed(42)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 10000
# Noise level
sigma = 0.01
pytorch_device_setting()
SILENT = False
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=0.1)
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=0.1)
#############################


from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
train_Y = pd.DataFrame(train_y)

test_X = pd.DataFrame(test_X.reshape(10000, 28 * 28))
test_Y = pd.DataFrame(test_y)

train_X = train_X[:n_samples]
train_Y = train_Y[:n_samples]
test_X = test_X[:n_samples]
test_Y = test_Y[:n_samples]

train_X = torch.from_numpy(train_X.values).float()
train_Y = torch.from_numpy(train_Y.values).long().squeeze() # squeeze for compatibility with loss function
test_X = torch.from_numpy(test_X.values).float()
test_Y = torch.from_numpy(test_Y.values).long().squeeze()  # squeeze for compatibility with loss function

if __name__ == '__main__':
    # config of the architecture:
    input_size = 28 * 28
    hidden_sizes = [200, 200]
    output_size = 10
    biases = [True, True, True]
    activation_functions = [F.relu, F.relu]
    dropout = 0.4
    epochs = 200
    batch_size = 2000
    optimiser = torch.optim.SGD
    criterion = nn.CrossEntropyLoss()
    dict_optimiser = {"lr": 0.001, "weight_decay": 0.01}

    parameters_for_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                                criterion=criterion, optimiser=optimiser,
                                                dict_params_optimiser=dict_optimiser)
    parametrized_NN = factory_parametrised_FC_NN(input_size=input_size, list_hidden_sizes=hidden_sizes,
                                                 output_size=output_size,
                                                 list_biases=biases, activation_functions=activation_functions,
                                                 dropout=dropout, predict_fct=lambda out: torch.max(out, 1)[1])

    test_nn_kfold_train.test(train_X, train_Y, parametrized_NN, parameters_for_training,
                             test_X, test_Y, early_stop_train, early_stop_valid,
                             SILENT,
                             compute_accuracy=True, plot_xx=None, plot_yy=None, plot_yy_noisy=None)
