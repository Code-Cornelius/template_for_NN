from priv_lib_plot import APlot

import test_nn_kfold_train

import torch
from torch import nn
import numpy as np
import pandas as pd

from plot.NN_plot_history import nn_plot_train_loss_acc
from plot.NN_plots import nn_plot_prediction_vs_true, nn_print_errors
from src.Neural_Network.Fully_connected_NN import factory_parametrised_FC_NN
from src.train.NNTrainParameters import NNTrainParameters
from src.Neural_Network.NN_fcts import pytorch_device_setting
from src.training_stopper.Early_stopper_training import Early_stopper_training
from src.training_stopper.Early_stopper_validation import Early_stopper_validation
from train.NN_kfold_training import nn_kfold_train

# set seed for pytorch.
torch.manual_seed(42)
np.random.seed(42)


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 2000
# Noise level
sigma = 0.01
device = pytorch_device_setting('not_cpu_please')
SILENT = False
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=0.01)
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=0.01)
early_stoppers = (early_stop_train, early_stop_valid)
metrics = ()
#############################

# exact grid
plot_xx = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
plot_yy = exact_solution(plot_xx).reshape(-1, )
plot_yy_noisy = (exact_solution(plot_xx) + sigma * torch.randn(plot_xx.shape)).reshape(-1, )

# random points for training
xx = 2 * np.pi * torch.rand((n_samples, 1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

# slicing:
training_size = int(90. / 100. * n_samples)
train_X = xx[:training_size, :]
train_Y = yy[:training_size, :]

testing_X = xx[training_size:, :]
testing_Y = yy[training_size:, :]
##### end data

if __name__ == '__main__':
    # config of the architecture:
    input_size = 1
    hidden_sizes = [20, 50, 20]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.relu]
    dropout = 0.
    epochs = 7500
    batch_size = 200
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction = 'sum')

    dict_optimiser = {"lr": 0.001, "weight_decay": 0.0000001}
    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optimiser=optimiser,
                                       dict_params_optimiser=dict_optimiser,
                                       metrics=metrics)

    parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size, param_list_hidden_sizes=hidden_sizes,
                                                 param_output_size=output_size, param_list_biases=biases,
                                                 param_activation_functions=activation_functions, param_dropout=dropout,
                                                 param_predict_fct=None)

    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, parametrized_NN,
                                                      parameters_training=param_training,
                                                      early_stoppers=early_stoppers, nb_split=1,
                                                      shuffle_kfold=True,
                                                      percent_validation_for_1_fold=10,
                                                      silent=False)
    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(history, flag_valid=True, log_axis_for_loss= True, best_epoch_of_NN=best_epoch_of_NN)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)
    APlot.show_plot()