import numpy as np
import pandas as pd
import torch
from priv_lib_plot import APlot
from priv_lib_util.tools.src.benchmarking import benchmark
from torch import nn
from tqdm import tqdm

from nn_classes.estimator.estim_benchmark_perf_nn_sizes import Estim_benchmark_perf_nn_sizes, \
    Plot_evol_benchmark_perf_nn_sizes
from nn_train.kfold_training import nn_kfold_train
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import set_seeds, pytorch_device_setting

# set seed for pytorch.

set_seeds(42)


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)


# 80% is taken for training
# batch is 1/4 of total training size.
sizes_samples = [50, 100, 200, 400, 1600, 2800, 3600,
                 6400, 8000, 12800, 16000, 21000, 25600]
# only divisble by 5 number in order to be ok for splitting sizes.

sizes_model = [10, 20, 40, 80, 160, 320, 500, 1000]

processing_units = ['gpu', 'cpu']


def benchmark_and_save(estim, input_size, type_pu, size_model, **kwargs):
    time = benchmark(nn_kfold_train, number_of_rep=1, silent_benchmark=True, **kwargs) / 100
    # divide by 100 bc we compute 100 epoch.
    time_dict = {"Input Size": input_size,
                 "Processing Unit": type_pu,
                 "Model Size": size_model,
                 "Comput. Time": [time]}
    estim.append(pd.DataFrame(time_dict))
    return


estim_bench = Estim_benchmark_perf_nn_sizes()

for size_sample in tqdm(sizes_samples):
    for PU in processing_units:
        for size_model in sizes_model:
            sigma = 0.0  # Noise level
            device = pytorch_device_setting(PU, True)
            SILENT = False
            xx = 2 * np.pi * torch.rand((size_sample, 1))
            yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

            training_size = int(80. / 100. * size_sample)
            train_X = xx[:training_size, :]
            train_Y = yy[:training_size, :]

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
            benchmark_and_save(estim_bench, size_sample, PU, size_model,
                               data_train_X=train_X, data_train_Y=train_Y,
                               Model_NN=parametrized_NN, param_train=param_training,
                               nb_split=1, shuffle_kfold=False,
                               percent_val_for_1_fold=0, silent=True)

plot_evol_estim = Plot_evol_benchmark_perf_nn_sizes(estim_bench)

plot_evol_estim.draw(column_name_draw='Comput. Time', envelope_flag=False,
                     separators_plot=["Processing Unit"], separator_colour='Model Size',
                     save_plot=False, dict_plot_for_main_line={})

APlot.show_plot()
