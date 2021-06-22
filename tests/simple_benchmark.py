import numpy as np
import torch
from priv_lib_plot import APlot
from torch import nn
from tqdm import tqdm

from nn_classes.estimator.plot_evol_history import Plot_evol_history
from nn_train.kfold_training import nn_kfold_train
from plot.nn_plots import nn_plot_prediction_vs_true, nn_errors_compute_mean
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_classes.metric.metric import Metric
from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import set_seeds, pytorch_device_setting

# set seed for pytorch.
set_seeds(42)


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


# 80% is taken for training
# batch is 1/4 of total training size.
sizes_samples = [400, 800, 1600, 2800, 3600,
                 6400, 8000, 11000, 12800, 15000,
                 18000, 21000, 25600]
# only divisble by 5 number in order to be ok for splitting sizes.

sizes_model = [5, 10, 20, 40, 80,
               120, 200, 400, 500, 1000]

computation_unit = ['gpu', 'cpu']

for size_sample in tqdm(sizes_samples):
    for CU in computation_unit:
        for size_model in sizes_model:
            ############################## GLOBAL PARAMETERS
            sigma = 0.0  # Noise level
            device = pytorch_device_setting(CU, True)
            SILENT = False
            ############################# DATA CREATION
            # random points for training
            xx = 2 * np.pi * torch.rand((size_sample, 1))
            yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

            # slicing:
            training_size = int(80. / 100. * size_sample)
            train_X = xx[:training_size, :]
            train_Y = yy[:training_size, :]
            ##### end data

            # config of the architecture:
            input_size = 1
            hidden_sizes = [size_model, size_model, size_model]
            output_size = 1
            biases = [True, True, True, True]
            activation_functions = [torch.tanh, torch.tanh, torch.relu]
            dropout = 0.
            epochs = 100
            batch_size = training_size // 4
            optimiser = torch.optim.Adam
            criterion = nn.MSELoss(reduction='sum')
            dict_optimiser = {"lr": 0.0005, "weight_decay": 1E-7}
            optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)
            param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                               criterion=criterion, optim_wrapper=optim_wrapper)
            parametrized_NN = factory_parametrised_FC_NN(param_input_size=input_size,
                                                         param_list_hidden_sizes=hidden_sizes,
                                                         param_output_size=output_size, param_list_biases=biases,
                                                         param_activation_functions=activation_functions,
                                                         param_dropout=dropout,
                                                         param_predict_fct=None)

            (net, estimator_history) = nn_kfold_train(train_X, train_Y, parametrized_NN, param_train=param_training,
                                                      nb_split=1, shuffle_kfold=False,
                                                      percent_val_for_1_fold=0, silent=True)


