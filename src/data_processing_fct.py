import torch
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from priv_lib_util.tools.src.function_writer import list_of_dicts_to_json


def create_input_sequences(input_data, lookback_window, lookforward_window=1,
                           time_series_dim=1, batch_first=True, silent=False):
    """

    Args:
        input_data (pytorch tensor):
        lookback_window (int):
        lookforward_window (int):
        time_series_dim (int):
        batch_first (bool):
        silent (bool):

    Returns:

    References :
        from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

    """
    L = len(input_data)
    nb_of_data = L - lookback_window - lookforward_window + 1

    assert lookback_window < L, f"lookback window is bigger than data. Window size : {lookback_window}, Data length : {L}."
    assert lookforward_window < L, f"lookforward window is bigger than data. Window size : {lookback_window}, Data length : {L}."

    if batch_first: # specifies how to take the input
        data_X = torch.zeros(nb_of_data, lookback_window, time_series_dim)
        data_Y = torch.zeros(nb_of_data, lookforward_window, time_series_dim)

        for i in tqdm(range(nb_of_data), disable=silent):
            data_X[i, :, :] = input_data[i:i + lookback_window].view(lookback_window, time_series_dim)
            data_Y[i, :,:] = input_data[i + lookback_window: i + lookback_window + lookforward_window].view(lookforward_window,time_series_dim)
        return data_X, data_Y

    else:
        data_X = torch.zeros(lookback_window, nb_of_data, time_series_dim)
        data_Y = torch.zeros(nb_of_data, time_series_dim, time_series_dim)

        for i in tqdm(range(nb_of_data), disable=silent):
            data_X[:, i, :] = input_data[i:i + lookback_window].view(lookback_window, time_series_dim)
            data_Y[:, i] = input_data[i + lookback_window: i + lookback_window + lookforward_window]
        return data_X, data_Y


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
