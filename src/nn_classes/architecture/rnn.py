from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from priv_lib_error import Error_type_setter

from src.nn_classes.architecture.savable_net import Savable_net


class RNN(Savable_net, metaclass=ABCMeta):
    def __init__(self):
        assert self.nb_output_consider <= self.input_time_series_len, \
            "The nb of output to consider (h_n, h_n-1...) needs to be smaller than the sequence length."
        super().__init__(predict_fct=None)  # predict is identity

        self.nb_directions = int(self.bidirectional) + 1

        self.stacked_rnn = self.rnn_class(self.input_dim, self.hidden_size,
                                          num_layers=self.num_layers,
                                          dropout=self.dropout,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)

        self.linear_layer = nn.Linear(self.hidden_size * self.nb_directions * self.nb_output_consider,
                                      self.hidden_FC)
        self.linear_layer_2 = nn.Linear(self.hidden_FC, self.output_dim * self.output_time_series_len)

    def forward(self, time_series):
        """
        Args:
            time_series: shape L,N, input_dim

        Returns:
        """
        batch_size = 1, time_series.shape[0], 1
        h0 = self.get_hidden_states(batch_size)
        out, _ = self.stacked_rnn(time_series, h0)  # shape of out is  N,L,Hidden_size * nb_direction

        if self.bidirectional:
            out = torch.cat((out[:, -self.nb_output_consider:, :self.hidden_size],
                             out[:, :self.nb_output_consider, self.hidden_size:]), 1)
            # this is where the output lies. We take nb_output elements. Meaning the h_n, h_n-1...
            # the shape of out at this stage is (N,  nb_output_consider, Hidden_size * nb_direction)

            # we do this because when the output is bidirectional, one should consider different outputs.

            # the first item is the uni direct, on top of it is stacked the other dirctn, whose first elmnts are taken.
        else:
            out = out[:, -self.nb_output_consider:, :self.hidden_size]

        out = out.view(-1, self.hidden_size * self.nb_directions * self.nb_output_consider)
        # squeezing the two last dimensions into one, for input to FC layer.

        out = self.linear_layer(out)
        out = self.activation_fct(out)
        out = self.linear_layer_2(out)
        return out.view(-1, self.output_time_series_len, self.output_dim)
        # batch size, dim time series output, dim output

    # section ######################################################################
    #  #############################################################################
    # SETTERS GETTERS

    @property
    @abstractmethod
    def input_dim(self):
        return self._input_dim

    @property
    @abstractmethod
    def output_dim(self):
        return self._output_dim

    @property
    @abstractmethod
    def hidden_size(self):
        return self._hidden_size

    @property
    @abstractmethod
    def bidirectional(self):
        return self._bidirectional

    @property
    @abstractmethod
    def num_layers(self):
        return self._num_layers

    @property
    @abstractmethod  # ABSTRACT FIELD
    def dropout(self):
        return self._dropout

    @property
    @abstractmethod
    def input_time_series_len(self):
        return self._input_time_series_len

    @property
    @abstractmethod
    def output_time_series_len(self):
        return self._output_time_series_len

    @property
    @abstractmethod
    def nb_output_consider(self):
        return self._nb_output_consider

    @property
    @abstractmethod
    def rnn_class(self):
        return self._nn_class

