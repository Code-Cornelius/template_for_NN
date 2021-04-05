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


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 1000
# Noise level
sigma = 0.01
pytorch_device_setting()
SILENT = False
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=0.1)
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=0.1)
#############################
plot_xx = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
plot_yy = exact_solution(plot_xx).reshape(-1, )
plot_yy_noisy = (exact_solution(plot_xx) + sigma * torch.randn(plot_xx.shape)).reshape(-1, )

xx = 2 * np.pi * torch.rand((n_samples, 1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

training_size = int(90. / 100. * n_samples)
train_X = xx[:training_size, :]
train_Y = yy[:training_size, :]

testing_X = xx[training_size:, :]
testing_Y = yy[training_size:, :]

if __name__ == '__main__':
    # config of the architecture:
    input_size = 1
    hidden_sizes = [20, 20, 20]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.relu]
    dropout = 0.
    epochs = 400
    batch_size = 200
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss()

    dict_optimiser = {"lr": 0.01, "weight_decay": 0.0000001}
    parameters_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                            criterion=criterion, optimiser=optimiser,
                                            dict_params_optimiser=dict_optimiser)
    parametrized_NN = factory_parametrised_FC_NN(input_size=input_size, list_hidden_sizes=hidden_sizes,
                                                 output_size=output_size,
                                                 list_biases=biases, activation_functions=activation_functions,
                                                 dropout=dropout, predict_fct=None)

    test_nn_kfold_train.test(train_X, train_Y, parametrized_NN, parameters_training, testing_X, testing_Y,
         early_stop_train, early_stop_valid, SILENT, compute_accuracy = False, plot_xx = plot_xx, plot_yy = plot_yy,
                             plot_yy_noisy = plot_yy_noisy)
