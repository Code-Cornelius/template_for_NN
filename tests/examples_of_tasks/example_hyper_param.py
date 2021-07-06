from priv_lib_util.tools.src.function_dict import parameter_product

from nn_classes.estimator.history.relplot_history import Relplot_history
from nn_classes.estimator.hyper_parameters.estim_hyper_param import Estim_hyper_param
from nn_classes.estimator.hyper_parameters.distplot_hyper_param import Distplot_hyper_param
from src.nn_classes.optim_wrapper import Optim_wrapper
from priv_lib_plot import APlot

import torch
from torch import nn
import numpy as np

from plot.nn_plots import nn_plot_prediction_vs_true, nn_errors_compute_mean
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import set_seeds, pytorch_device_setting
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from nn_train.kfold_training import nn_kfold_train
from src.nn_classes.metric.metric import Metric


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


def L4loss(net, xx, yy):
    return torch.norm(net.nn_predict(xx) - yy, 4)

L4metric = Metric('L4', L4loss)
metrics = (L4metric,)

############################## GLOBAL PARAMETERS
n_samples = 2000  # Number of training samples
sigma = 0.01  # Noise level
device = pytorch_device_setting('cpu')
SILENT = False
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=-int(1E-6))
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=-int(1E-6))
early_stoppers = (early_stop_train, early_stop_valid)
############################# DATA CREATION
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

params_options = {
    "seed": [42],
    "lr": [0.001, 0.0001, 0.01, 0.0005, 0.005],
    "dropout": [0., 0.1, 0.2, 0.25, 0.05],
    "hidden_sizes": [[20, 50, 20], [30, 40, 30], [10, 10, 10], [20, 80, 20], [10, 30, 20]]
}

hyper_params = parameter_product(params_options)

def config_architecture(params):
    # config of the architecture:
    input_size = 1
    hidden_sizes = params["hidden_sizes"]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.relu]
    dropout = params["dropout"]
    epochs = 7500
    batch_size = 200
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": params["lr"], "weight_decay": 0.0000001}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)
    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)
    Class_Parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                       param_list_hidden_sizes=hidden_sizes,
                                                       param_output_size=output_size, param_list_biases=biases,
                                                       param_activation_functions=activation_functions,
                                                       param_dropout=dropout,
                                                       param_predict_fct=None)

    return param_training, Class_Parametrized_NN



def generate_estims_history():
    for i, params in enumerate(hyper_params):
        # set seed for pytorch.
        set_seeds(params["seed"])

        param_training, Class_Parametrized_NN = config_architecture(params)

        (net, estimator_history) = nn_kfold_train(train_X,
                                                  train_Y,
                                                  Class_Parametrized_NN,
                                                  param_train=param_training,
                                                  early_stoppers=early_stoppers,
                                                  nb_split=9,
                                                  shuffle_kfold=True,
                                                  percent_val_for_1_fold=10,
                                                  silent=False,
                                                  hyper_param=params)

        estimator_history.to_json(path=f"sin_estim_history/estim_{i}.json")

NEW_DATASET = False

if __name__ == '__main__':

    if NEW_DATASET:
        generate_estims_history()
        estim = Estim_hyper_param.from_folder(path="sin_estim_history", metric_name="loss_validation")
        estim.to_csv("test_estim_hyper_param.csv")
    else:
        estim = Estim_hyper_param.from_csv("test_estim_hyper_param.csv")

    plot_hist_estim = Distplot_hyper_param(estim)
    plot_hist_estim.hist(column_name_draw='loss_validation', separators_plot=['dropout'], hue='lr',
                         palette='PuOr', bins=5,
                         binrange=None, stat='count', multiple="layer", kde=True, path_save_plot=None)

    APlot.show_plot()

