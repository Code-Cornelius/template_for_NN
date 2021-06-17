import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#TODO USE WINDOW CREATOR
# from data_processing_fct import create_input_sequences
from nn_classes.estimator.estim_history import Estim_history
from src.nn_classes.architecture.gru import factory_parametrised_GRU
from src.plot.nn_plot_history import nn_plot_train_loss_acc
from src.plot.nn_plots import nn_plot_prediction_vs_true, nn_print_errors
from src.nn_classes.architecture.fully_connected import factory_parametrised_FC_NN
from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import pytorch_device_setting, set_seeds
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from src.nn_train.kfold_training import nn_kfold_train, train_kfold_a_fold_after_split, create_history_kfold
from src.nn_classes.architecture.lstm import factory_parametrised_LSTM

from priv_lib_plot import APlot

# set seed for pytorch.
set_seeds(42)

############### PARAMETERS:
time_series_len = lookback_window = 500
###############

import pandas as pd

validation_and_testing_data_size = lookback_window
# Import Data
flight_data = pd.read_csv("../research_on_time_series_forecasting/daily-min-temperatures.csv")
print(flight_data.head())
data = flight_data['Temp'].values.astype(float)

train_data = data[:-validation_and_testing_data_size]

minimax = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = torch.FloatTensor(minimax.fit_transform(train_data.reshape(-1, 1))).view(-1)
# : the reshape in order to fit and transform
testing_data = data[-2 * validation_and_testing_data_size:].reshape(-1, 1)
testing_data_normalised = torch.FloatTensor(minimax.transform(testing_data)).view(1, lookback_window * 2, 1)
testing_data = torch.FloatTensor(testing_data).view(1, lookback_window * 2, 1)


def inverse_transform(arr):
    return minimax.inverse_transform(np.array(arr).reshape(-1, 1)).reshape(-1)


############################## GLOBAL PARAMETERS
device = pytorch_device_setting('gpu')
SILENT = False
early_stop_train = Early_stopper_training(patience=400, silent=SILENT, delta=-int(1E-2))
early_stop_valid = Early_stopper_validation(patience=400, silent=SILENT, delta=-int(1E-2))
early_stoppers = (early_stop_train, early_stop_valid)
metrics = ()
#############################
data_training_X, data_training_Y = create_input_sequences(train_data_normalized, lookback_window)
indices_train = torch.arange(len(data_training_X) - validation_and_testing_data_size)
indices_valid = torch.arange(len(data_training_X) - validation_and_testing_data_size, len(data_training_X))
print("shape of training : ", data_training_Y.shape)

if __name__ == '__main__':
    # config of the architecture:
    input_size = 1
    num_layers = 5
    bidirectional = False
    hidden_size = 150
    output_size = 1
    dropout = 0.
    epochs = 80
    batch_size = 100

    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": 0.00005, "weight_decay": 0.0000001}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)

    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)

    parametrized_NN = factory_parametrised_LSTM(input_size=input_size, num_layers=num_layers,
                                                bidirectional=bidirectional, time_series_len=time_series_len,
                                                hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                                                activation_fct=nn.CELU(), hidden_FC=50)

    estimator_history = Estim_history(metric_names=[], validation=True)
    net, _ = train_kfold_a_fold_after_split(data_training_X, data_training_Y, indices_train,
                                            indices_valid, parametrized_NN, param_training, estimator_history,
                                            early_stoppers=early_stoppers)

    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(estimator_history, flag_valid=True, log_axis_for_loss=True, key_for_second_axis_plot=None,
                           log_axis_for_second_axis=True)

    ########## prediction :
    # prediction by looking at the data we know about
    train_target = data_training_Y
    train_prediction = net.nn_predict(data_training_X)

    # prediction, test set. Corresponds to predicting the black line.
    test = testing_data_normalised
    print("Prediction over test data: ", inverse_transform(test))
    test_prediction = [0] * validation_and_testing_data_size
    for i in range(validation_and_testing_data_size):
        test_prediction[i] = net.nn_predict(test[:, i:lookback_window + i, :])

    # prediction of unknown data by starting with black line.
    unknown = testing_data_normalised[:, validation_and_testing_data_size:, :].repeat(1, 2, 1)
    print("Prediction over unknown data: ", inverse_transform(unknown))
    # : the last data + a copy to store prediction
    for i in range(validation_and_testing_data_size):
        unknown[:, i + lookback_window, :] = net.nn_predict(unknown[:, i:lookback_window + i, :])
        # : replacing the second copy by predictions
    unknwon_prediction = unknown[:, -validation_and_testing_data_size:, :]

    months_total = np.arange(0, len(data), 1)
    months_train = np.arange(0, len(train_data), 1)
    months_train_prediction = np.arange(lookback_window, len(train_prediction) + lookback_window, 1)
    months_test = np.arange(len(data_training_Y) + lookback_window,
                            len(data_training_Y) + lookback_window + validation_and_testing_data_size, 1)
    months_forecast = np.arange(len(data_training_Y) + lookback_window + validation_and_testing_data_size,
                                len(data_training_Y) + lookback_window + validation_and_testing_data_size * 2, 1)

    aplot = APlot()
    dict_ax = {'title': 'forecasting', 'xlabel': 'month', 'ylabel': 'passenger'}
    dict_plot_param = {'label': 'Data for Testing', 'color': 'black', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_total, data, dict_ax=dict_ax, dict_plot_param=dict_plot_param)
    dict_plot_param = {'label': 'Data Known at Training Time', 'color': 'gray', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_train, train_data, dict_ax=dict_ax, dict_plot_param=dict_plot_param)
    APlot.show_and_continue()

    dict_plot_param = {'label': 'Prediction over Training', 'color': 'royalblue', 'linestyle': '--', 'linewidth': 2}
    aplot.uni_plot(0, months_train_prediction, inverse_transform(train_prediction), dict_plot_param=dict_plot_param)
    dict_plot_param = {'label': 'Prediction over Test Set', 'color': 'r', 'linestyle': '--', 'linewidth': 1}
    aplot.uni_plot(0, months_test, inverse_transform(test_prediction), dict_plot_param=dict_plot_param)
    dict_plot_param = {'label': 'Prediction of Future Unknown Set', 'color': 'g', 'linestyle': '--', 'linewidth': 1}
    aplot.uni_plot(0, months_forecast, inverse_transform(unknwon_prediction), dict_plot_param=dict_plot_param)
    aplot.show_legend()
    APlot.show_plot()