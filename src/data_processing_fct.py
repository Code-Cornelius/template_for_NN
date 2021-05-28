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


def add_column_cyclical_features(df, col_name_time, period, start_num=0):
    """
    Semantics:
        In order to incorporate cyclicity in input data, one can add the sin/cos of the time data (e.g.).

    Args:
        df: pandas dataframe.
        col_name_time (str):  name of the column where the cyclicity is computed from.
        period: period in terms of values from the col_name_time.
        start_num (float): starting value of the cyclicity. Default = 0.

    Pre-condition:
        df's col_name_time exists.

    Post-condition:
        df's col_name_time is removed.
        df's  'sin_{col_name_time}' and 'cos_{col_name_time}' are created.

    Returns:
        The new dataframe that needs to be reassigned.
    """
    values = 2 * np.pi * (df[col_name_time] - start_num) / period
    kwargs = {f'sin_{col_name_time}': lambda x: np.sin(values),
              f'cos_{col_name_time}': lambda x: np.cos(values)}
    return df.assign(**kwargs).drop(columns=[col_name_time])
