import sklearn
from priv_lib_plot import APlot

from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_classes.metric.metric import Metric

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

from plot.nn_plot_history import nn_plot_train_loss_acc
from plot.nn_plots import nn_plot_prediction_vs_true, nn_print_errors
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import set_seeds, pytorch_device_setting
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from nn_train.kfold_training import nn_kfold_train

# set seed for pytorch.
set_seeds(42)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 10000
# Noise level
sigma = 0.01
device = pytorch_device_setting('not_cpu_please')
SILENT = False
early_stop_train = Early_stopper_training(patience=200, silent=SILENT, delta=-0.05)
early_stop_valid = Early_stopper_validation(patience=200, silent=SILENT, delta=-0.05)
early_stoppers_train = (early_stop_train,)
early_stoppers_valid = (early_stop_valid,)
early_stoppers = (early_stop_train, early_stop_valid)

accuracy_wrapper = lambda net, xx, yy: sklearn.metrics.accuracy_score(net.nn_predict_ans2cpu(xx),
                                                                      yy.reshape(-1, 1).to('cpu'),
                                                                      normalize=False
                                                                      )
accuracy_metric = Metric(name="accuracy", function=accuracy_wrapper)
metrics = (accuracy_metric,)
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
train_Y = torch.from_numpy(train_Y.values).long().squeeze()  # squeeze for compatibility with loss function
test_X = torch.from_numpy(test_X.values).float()
test_Y = torch.from_numpy(test_Y.values).long().squeeze()  # squeeze for compatibility with loss function

if __name__ == '__main__':
    # config of the architecture:
    input_size = 28 * 28
    hidden_sizes = [100]
    output_size = 10
    biases = [True, True]
    activation_functions = [F.relu]
    dropout = 0.2
    epochs = 1000
    batch_size = 1000
    optimiser = torch.optim.SGD
    criterion = nn.CrossEntropyLoss(reduction='sum')
    dict_optimiser = {"lr": 0.0000005, "weight_decay": 0.00001}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)

    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)
    Class_Parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                       param_list_hidden_sizes=hidden_sizes,
                                                       param_output_size=output_size, param_list_biases=biases,
                                                       param_activation_functions=activation_functions,
                                                       param_dropout=dropout,
                                                       param_predict_fct=lambda out: torch.max(out, 1)[1])

    (net, estimator_history) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN, param_train=param_training,
                                              early_stoppers=early_stoppers, nb_split=1, shuffle_kfold=True,
                                              percent_val_for_1_fold=10, silent=False)

    nn_plot_train_loss_acc(estimator_history, flag_valid=True, log_axis_for_loss=True,
                           key_for_second_axis_plot="accuracy", log_axis_for_second_axis=False)
    APlot.show_plot()
