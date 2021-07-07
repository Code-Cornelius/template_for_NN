import numpy as np
import torch
import os
from priv_lib_plot import APlot
from priv_lib_util.tools.src.function_dict import parameter_product
from torch import nn
from tqdm import tqdm

from nn_classes.estimator.hyper_parameters.distplot_hyper_param import Distplot_hyper_param
from nn_classes.estimator.hyper_parameters.estim_hyper_param import Estim_hyper_param
from nn_train.kfold_training import nn_kfold_train
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_classes.metric.metric import Metric
from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import set_seeds, pytorch_device_setting


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
    "seed": [42, 124, 300],
    "lr": [0.01, 0.001, 0.0001],
    "dropout": [0., 0.05, 0.1, 0.25],
    "hidden_sizes": [[2,4,2], [4, 8, 4], [8, 16, 8], [8, 32, 8], [16, 64, 32], [64, 128, 64]]
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
    batch_size = 1000
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
    for i, params in enumerate(tqdm(hyper_params)):
        # set seed for pytorch.
        set_seeds(params["seed"])

        param_training, Class_Parametrized_NN = config_architecture(params)

        (net, estimator_history) = nn_kfold_train(train_X,
                                                  train_Y,
                                                  Class_Parametrized_NN,
                                                  param_train=param_training,
                                                  early_stoppers=early_stoppers,
                                                  nb_split=1,
                                                  shuffle_kfold=True,
                                                  percent_val_for_1_fold=20,
                                                  silent=True,
                                                  hyper_param=params)

        estimator_history.to_json(path=f"sin_estim_history/estim_{i}.json")


NEW_DATASET = False

def cleanup(folder_path):
    for file in os.listdir(folder_path):
        if file.startswith("estim") and file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

if __name__ == '__main__':
    FOLDER_PATH = "sin_estim_history"
    # todo make hyper param init for a list of estims
    if NEW_DATASET:
        generate_estims_history()
        estim = Estim_hyper_param.from_folder(path=FOLDER_PATH, metric_name="loss_validation")
        cleanup(FOLDER_PATH)
        estim.to_csv("test_estim_hyper_param.csv")
    if not NEW_DATASET:
        estim = Estim_hyper_param.from_csv("test_estim_hyper_param.csv")

        histplot_hyperparam = Distplot_hyper_param(estim)
        histplot_hyperparam.hist(column_name_draw='loss_validation', separators_plot=None, hue='hidden_sizes',
                                 palette='RdYlBu', bins=20,
                                 binrange=None, stat='count', multiple="stack", kde=False, path_save_plot=None)

        histplot_hyperparam.hist(column_name_draw='loss_validation', separators_plot=None, hue='seed',
                                 palette='RdYlBu', bins=20,
                                 binrange=None, stat='count', multiple="dodge", kde=True, path_save_plot=None)

        APlot.show_plot()

        # TODO 06/07/2021 nie_k:  how could i multiply all loses by 1000?