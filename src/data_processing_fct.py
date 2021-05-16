import torch
from tqdm import tqdm


def create_input_sequences(input_data, lookback_window, data_input_dim=1, output_dim=1, batch_first=True, silent = False):
    """

    Args:
        input_data (pytorch tensor):
        lookback_window (int):
        data_input_dim (int):
        output_dim (int):
        batch_first (bool):
        silent (bool):

    Returns:

    References :
        from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/?fbclid=IwAR17NoARUlBsBLzanKmyuvmCXfU6Rxc69T9BZpowXfSUSYQNEFzl2pfDhSo

    """
    L = len(input_data)
    nb_of_data = L - lookback_window

    assert lookback_window < L, f"lookback window is bigger than data. Window size : {lookback_window}, Data length : {L}."

    if batch_first:
        data_X = torch.zeros(nb_of_data, lookback_window, data_input_dim)
        data_Y = torch.zeros(nb_of_data, output_dim)

        for i in tqdm(range(nb_of_data), disable = silent):
            data_X[i, :, :] = input_data[i:i + lookback_window].view(lookback_window, data_input_dim)
            data_Y[i, :] = input_data[i + lookback_window]
        return data_X, data_Y
    else:
        data_X = torch.zeros(lookback_window, nb_of_data, data_input_dim)
        data_Y = torch.zeros(nb_of_data, output_dim)

        for i in tqdm(range(nb_of_data), disable=silent):
            data_X[:, i, :] = input_data[i:i + lookback_window]
            data_Y[:, i] = input_data[i + lookback_window]
        return data_X, data_Y
