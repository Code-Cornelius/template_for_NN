import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from priv_lib_plot import APlot
from sklearn.preprocessing import MinMaxScaler

from data_processing_fct import add_column_cyclical_features
from nn_classes.architecture.free_nn import factory_parametrised_Free_NN
from nn_classes.architecture.reshape import Reshape
from nn_classes.estimator.history.estim_history import Estim_history
from nn_classes.estimator.history.relplot_history import Relplot_history
from nn_classes.factory_parametrised_rnn import factory_parametrised_RNN
from nn_classes.windowcreator import Windowcreator
from src.nn_classes.architecture.lstm import LSTM
from src.nn_classes.optim_wrapper import Optim_wrapper
from src.nn_classes.training_stopper.Early_stopper_training import Early_stopper_training
from src.nn_classes.training_stopper.Early_stopper_validation import Early_stopper_validation
from src.nn_train.kfold_training import train_kfold_a_fold_after_split
from src.nn_train.nntrainparameters import NNTrainParameters
from src.util_training import pytorch_device_setting, set_seeds

# set seed for pytorch.
set_seeds(42)

# Import Data
TEMP_DATA = pd.read_csv("../../research_on_time_series_forecasting/daily_min_temperatures.csv")

############################## GLOBAL PARAMETERS
device = pytorch_device_setting('gpu')
SILENT = False
early_stop_train = Early_stopper_training(patience=400, silent=SILENT, delta=-int(1E-2))
early_stop_valid = Early_stopper_validation(patience=400, silent=SILENT, delta=-int(1E-2))
early_stoppers = (early_stop_train, early_stop_valid)
metrics = ()
##########################################  DATA PARAMETERS:
time_series_len = lookback_window = 365
lookforward_window = 365
nb_test_prediction = lookback_window // lookforward_window + 1
nb_unknown_prediction = 365 // lookforward_window
##########################################  main
if __name__ == '__main__':
    # config of the architecture:
    input_dim = 3
    num_layers = 4
    bidirectional = True
    hidden_size = 36
    output_dim = 1
    dropout = 0.1
    epochs = 80000
    batch_size = 1500
    hidden_FC = 128

    optimiser = torch.optim.Adam
    criterion = nn.MSELoss(reduction='sum')
    dict_optimiser = {"lr": 0.0005, "weight_decay": 1E-6}
    optim_wrapper = Optim_wrapper(optimiser, dict_optimiser)

    param_training = NNTrainParameters(batch_size=batch_size, epochs=epochs, device=device,
                                       criterion=criterion, optim_wrapper=optim_wrapper,
                                       metrics=metrics)

    seq_nn = [
        Reshape([-1, input_dim, lookback_window]),
        nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=11, stride=1, padding=0, dilation=1,
                  groups=1, bias=True),
        nn.AvgPool1d(kernel_size=9, stride=3, padding=1),
        nn.BatchNorm1d(input_dim),
        nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=0, dilation=1,
                  groups=1, bias=True),
        nn.MaxPool1d(kernel_size=5, stride=2, padding=1),
        nn.BatchNorm1d(input_dim),
        Reshape([-1, 57, input_dim]),
        (model := factory_parametrised_RNN(input_dim=input_dim, output_dim=output_dim,
                                           num_layers=num_layers, bidirectional=bidirectional,
                                           input_time_series_len=57,
                                           output_time_series_len=lookforward_window,
                                           nb_output_consider=40,
                                           hidden_size=hidden_size, dropout=dropout,
                                           Parent=LSTM, rnn_class=nn.LSTM)()),  # walrus operator
        Reshape([-1, model.output_len]),
        nn.Linear(model.output_len, 1000, bias=True),
        nn.CELU(),
        nn.Linear(1000, lookforward_window * output_dim, bias=True),
        Reshape([-1, lookforward_window, output_dim]),
    ]

    parametrized_NN = factory_parametrised_Free_NN(seq_nn)

    ########################################## DATA
    print(TEMP_DATA.head())
    TEMP_DATA['day'] = range(len(TEMP_DATA))
    TEMP_DATA = TEMP_DATA.drop(columns=["Date"])
    # add cyclicity
    PERIOD_CYCLE = 364.25
    TEMP_DATA = add_column_cyclical_features(TEMP_DATA, 'day', PERIOD_CYCLE)
    print(TEMP_DATA.head())

    if input_dim == 1:
        data = TEMP_DATA['Temperature'].values.astype(float).reshape(-1, 1)
    else:
        data = TEMP_DATA.values.astype(float)

    train_data = data[:-lookback_window].reshape(-1, input_dim)
    testing_data = data[-lookback_window:].reshape(-1, input_dim)

    minimax = MinMaxScaler(feature_range=(-1., 1.))
    train_data_normalized = torch.FloatTensor(minimax.fit_transform(train_data))
    testing_data_normalised = torch.FloatTensor(minimax.transform(testing_data))
    testing_data = torch.FloatTensor(testing_data)

    window = Windowcreator(input_dim=input_dim, output_dim=1, lookback_window=lookback_window,
                           lag_last_pred_fut=lookforward_window,
                           lookforward_window=lookforward_window, type_window="Moving")
    (data_training_X,
     data_training_Y) = window.create_input_sequences(train_data_normalized, train_data_normalized[:, 0].unsqueeze(1))

    indices_train = torch.arange(len(data_training_X) - lookback_window)
    indices_valid = torch.arange(len(data_training_X) - lookback_window, len(data_training_X))
    print("shape of training : ", data_training_Y.shape)
    print("data training : ", data_training_X)


    ##########################################

    def inverse_transform(arr):
        return minimax.inverse_transform(np.array(arr).reshape(-1, input_dim))


    ##########################################  TRAINING

    estimator_history = Estim_history(metric_names=[], validation=True)
    net, _ = train_kfold_a_fold_after_split(data_training_X, data_training_Y, indices_train, indices_valid,
                                            parametrized_NN, param_training, estimator_history,
                                            early_stoppers=early_stoppers)
    net.to(torch.device('cpu'))
    history_plot = Relplot_history(estimator_history)
    history_plot.lineplot(log_axis_for_loss=True)
    history_plot.draw_two_metrics_same_plot(key_for_second_axis_plot=None,
                                            log_axis_for_loss=True, log_axis_for_second_axis=True)


    class Adaptor_output(object):
        def __init__(self, incrm, start_num, period):
            self.incrm = incrm
            self.start_num = start_num
            self.period = period

        def __call__(self, arr):
            times = np.arange(self.incrm, self.incrm + lookforward_window)
            cos_val = np.sin(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
            sin_val = np.cos(2 * np.pi * (times - self.start_num) / self.period).reshape(-1, 1)
            self.incrm += lookforward_window
            res = np.concatenate((arr.reshape(-1, 1), cos_val, sin_val), axis=1).reshape(1, -1, 3)
            return torch.tensor(res, dtype=torch.float32)


    ##########################################  prediction TESTING :
    # prediction by looking at the data we know about
    if input_dim > 1:
        increase_data_for_pred = Adaptor_output(0, 0, PERIOD_CYCLE)  # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    train_prediction = window.prediction_over_training_data(net, train_data_normalized, increase_data_for_pred,
                                                            device=device)

    if input_dim > 1:
        increase_data_for_pred = Adaptor_output(train_data_normalized.shape[0], 0, PERIOD_CYCLE)
        # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    ##########################################  prediction unknown set. Corresponds to predicting the black line.
    test_prediction = window.prediction_recurrent(net, train_data_normalized[-lookback_window:],
                                                  nb_test_prediction, increase_data_for_pred, device='cpu')

    ##########################################  prediction of TESTING unknown data by starting with black line.
    if input_dim > 1:
        increase_data_for_pred = Adaptor_output(train_data_normalized.shape[0] + testing_data_normalised.shape[0],
                                                0, PERIOD_CYCLE)  # 0 initial and 12 month for a period.
    else:
        increase_data_for_pred = None
    unknwon_prediction = window.prediction_recurrent(net, testing_data_normalised[-lookback_window:, :],
                                                     nb_unknown_prediction, increase_data_for_pred, device='cpu')

    # x-axis data for plotting
    months_total = np.arange(0, len(data), 1)  # data for testing
    months_train = np.arange(0, len(train_data), 1)  # data for training
    months_train_prediction = np.arange(lookback_window, train_prediction.shape[1] + lookback_window, 1)
    length_training = len(data_training_Y) + lookback_window + lookforward_window - 1
    months_test = np.arange(length_training,
                            length_training + nb_test_prediction * lookforward_window, 1)
    months_forecast = np.arange(length_training + lookback_window,
                                length_training + lookback_window + lookforward_window * nb_unknown_prediction,
                                1)

    aplot = APlot()
    xlabel = 'Day'
    ylabel = 'temperature in Celcius'
    dict_ax = {'title': 'Data and Prediction during Training and over Non Observed Data.', 'xlabel': xlabel,
               'ylabel': ylabel}
    dict_plot_param = {'label': 'Data for Testing', 'color': 'black', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_total, data[:, 0], dict_ax=dict_ax, dict_plot_param=dict_plot_param)
    dict_plot_param = {'label': 'Data Known at Training Time', 'color': 'gray', 'linestyle': '-', 'linewidth': 3}
    aplot.uni_plot(0, months_train, train_data[:, 0], dict_ax=dict_ax, dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction over Training', 'color': 'royalblue', 'linestyle': '--', 'linewidth': 2}
    aplot.uni_plot(0, months_train_prediction, inverse_transform(train_prediction)[:, 0],
                   dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction over Test Set', 'color': 'r', 'linestyle': '--', 'linewidth': 1.5}
    aplot.uni_plot(0, months_test, inverse_transform(test_prediction)[:, 0], dict_plot_param=dict_plot_param)

    dict_plot_param = {'label': 'Prediction of Future Unknown Set', 'color': 'g', 'linestyle': '--', 'linewidth': 1.5}
    aplot.uni_plot(0, months_forecast, inverse_transform(unknwon_prediction)[:, 0], dict_plot_param=dict_plot_param)
    aplot.show_legend()
    APlot.show_plot()
