from abc import ABCMeta

import torch
import torch.nn as nn

from nn_classes.architecture.gru import GRU, factory_parametrised_GRU


class LSTM(GRU, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()  # predict is identity

        self.hidden_cell_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                      1,  # repeated later to have batch size
                                                      self.hidden_size),
                                          requires_grad=True)  # parameters are moved to device and learn.

    def get_stacked_in(self, batch_size):
        return self.hidden_state_0.repeat(batch_size), self.hidden_cell_0.repeat(batch_size)


def factory_parametrised_LSTM(input_dim=1, output_dim=1,
                              num_layers=1, bidirectional=False,
                              input_time_series_len=1, output_time_series_len=1,
                              nb_output_consider=1,
                              hidden_size=150, dropout=0.,
                              activation_fct=nn.CELU(), hidden_FC=64):

    # todo: can use args, kwargs to not pass all params again?
    return factory_parametrised_GRU(input_dim=input_dim,
                                    output_dim=output_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_time_series_len=input_time_series_len,
                                    output_time_series_len=output_time_series_len,
                                    nb_output_consider=nb_output_consider,
                                    hidden_size=hidden_size,
                                    dropout=dropout,
                                    activation_fct=activation_fct,
                                    hidden_FC=hidden_FC,
                                    nn_class=nn.LSTM,
                                    parent=LSTM)
