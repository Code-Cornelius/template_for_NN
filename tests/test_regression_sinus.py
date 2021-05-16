from src.nn_classes.optim_wrapper import Optim_wrapper
from priv_lib_plot import APlot

import test_nn_kfold_train

import torch
from torch import nn
import numpy as np
import pandas as pd

from plot.nn_plot_history import nn_plot_train_loss_acc
from plot.nn_plots import nn_plot_prediction_vs_true, nn_print_errors
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.train.nntrainparameters import NNTrainParameters
from src.nn_classes.architecture.nn_fcts import pytorch_device_setting, set_seeds
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from train.nn_kfold_training import nn_kfold_train

# set seed for pytorch.
set_seeds(42)


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
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=-int(1E-6))
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=-int(1E-6))
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

    def L4loss(net, xx, yy):
        return torch.norm(net.nn_predict(xx) - yy, 4)

    from src.nn_classes.metric.metric import Metric

    L4metric = Metric('L4', L4loss)
    metrics = (L4metric,)

    dict_optimiser = {"lr": 0.001, "weight_decay": 0.0000001}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)

    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)

    parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size, param_list_hidden_sizes=hidden_sizes,
                                                 param_output_size=output_size, param_list_biases=biases,
                                                 param_activation_functions=activation_functions, param_dropout=dropout,
                                                 param_predict_fct=None)

    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, parametrized_NN,
                                                      parameters_training=param_training, early_stoppers=early_stoppers,
                                                      nb_split=1, shuffle_kfold=True, percent_validation_for_1_fold=10,
                                                      silent=False)
    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(history, flag_valid=True, log_axis_for_loss= True, best_epoch_of_NN=best_epoch_of_NN,
                           key_for_second_axis_plot = 'L4', log_axis_for_second_axis = True)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)
    APlot.show_plot()