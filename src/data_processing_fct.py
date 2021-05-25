import torch
from tqdm import tqdm
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from priv_lib_util.tools.src.function_writer import list_of_dicts_to_json


# TODO
def save_data(net, history, name_config, number_run, Taskname, best_epoch_of_NN, root_dir,
              number_epoch_between_savings=1):
    """
    changes history !!! Changes the format (compressed), replaces numpy arrays by lists, and adds an entry 'best_epoch'.
    """
    PATH_RESULTS_RESULT = os.path.join(root_dir, Taskname, 'results')

    def linked_path(path):
        return os.path.join(PATH_RESULTS_RESULT, *path)

    path_save_history = linked_path(['history', name_config, f"history_{number_run}.json"])
    path_save_nets = linked_path(['nets', name_config, f"net_{number_run}.pth"])

    net.save_net(path_save_nets)
    # converting every numpy array to a list.
    for key, value in history.items():
        for key_loss, value_loss in value.items():
            value[key_loss] = value_loss[:, number_epoch_between_savings].tolist()
    history['best_epoch'] = best_epoch_of_NN

    # write a list of dicts into a JSON, we compress the data.
    list_of_dicts_to_json(parameter_options=history, file_name=path_save_history, compress=True)


def pipeline_scaling_minimax(df):
    minimax = MinMaxScaler(feature_range=(0, 1))
    minimax.fit(df)
    return minimax, minimax.transform(df)


def pipeline_scaling_normal(df):
    standar_normalis = StandardScaler()  # norm l2 and gives a N_0,1, on each column.
    standar_normalis.fit(df.values.reshape(-1, 1))
    return standar_normalis, standar_normalis.transform(df.values.reshape(-1, 1)).reshape(-1)


# TODO useful?
def read_list_of_ints_from_path(path):
    ans = []
    with open(path, "r") as file:
        for line in file:
            ans.append(float(line.strip()))
    return ans


class Windowcreator(object):
    def __init__(self, input_dim, output_dim,
                 lookforward_window,
                 lookback_window=0, lag_last_pred_fut=1,
                 type_window="Moving",
                 batch_first=True, silent=False):
        """
        Data pred is the same as input.
        References:
            https://www.tensorflow.org/tutorials/structured_data/time_series#2_split"""

        assert type_window == "Increasing" or type_window == "Moving", "Only two types supported."
        assert not (type_window == "Increasing" and lookback_window != 0), "Increasing so window ==0."
        assert not (type_window == "Moving" and lookback_window == 0), "Moving so window > 0."

        # Window parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lookback_window = lookback_window
        self.lookforward_window = lookforward_window
        self.type_window = type_window
        self.lag_last_pred_fut = lag_last_pred_fut

        self.batch_first = batch_first

        self.silent = silent

        # Parameters of the slices
        self.complete_window_data = self.lookback_window + self.lag_last_pred_fut

        self.input_slice = slice(0, self.lookback_window)
        self.input_indices = np.arange(self.complete_window_data)[self.input_slice]

        self.index_start_prediction = self.complete_window_data - self.lookforward_window
        self.slices_prediction = slice(self.index_start_prediction, None)  # None means to the end
        self.indices_prediction = np.arange(self.complete_window_data)[self.slices_prediction]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.complete_window_data}',
            f'X indices: {self.input_indices}',
            f'Y indices: {self.indices_prediction}'])

    def create_input_sequences(self, input_data, output_data):
        """

        Args:
            input_data (pytorch tensor): should be a N*M matrix, column is a time series.
            output_data (pytorch tensor): should be a N'*M' matrix, column is a time series.

        Returns:

        References :
            from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

        """
        L = len(input_data)  # WIP make sure it does what one expects.
        nb_of_data = L - self.complete_window_data + 1  # nb of data - the window, but there is always one data so +1.

        assert self.lookback_window < L, \
            f"lookback window is bigger than data. Window size : {self.lookback_window}, Data length : {L}."
        assert self.lookforward_window < L, \
            f"lookforward window is bigger than data. Window size : {self.lookback_window}, Data length : {L}."
        assert len(input_data) == len(output_data)

        if self.batch_first:  # specifies how to take the input
            data_X = torch.zeros(nb_of_data, self.lookback_window, self.input_dim)
            data_Y = torch.zeros(nb_of_data, self.lookforward_window, self.output_dim)

            for i in tqdm(range(nb_of_data), disable=self.silent):
                data_X[i, :, :] = input_data[i:i + self.lookback_window, :].view(self.lookback_window, self.input_dim)
                data_Y[i, :, :] = output_data[
                                  i + self.lookback_window: i + self.lookback_window +
                                                            self.lookforward_window, :].view(self.lookforward_window,
                                                                                             self.output_dim)
            return data_X, data_Y

        else:
            data_X = torch.zeros(self.lookback_window, nb_of_data, self.input_dim)
            data_Y = torch.zeros(nb_of_data, self.input_dim, self.output_dim)

            for i in tqdm(range(nb_of_data), disable=self.silent):
                data_X[:, i, :] = input_data[i:i + self.lookback_window, :].view(self.lookback_window, self.input_dim)
                data_Y[:, i, :] = output_data[i + self.lookback_window: i + self.lookback_window +
                                                                       self.lookforward_window, :]
            return data_X, data_Y
