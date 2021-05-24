from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from priv_lib_error import Error_type_setter

from src.nn_classes.architecture.savable_net import Savable_net


# Todo make sure there is input dim, output dim, input length, output length,
#  how many hn to consider for hidden layers.
class LSTM(Savable_net, metaclass=ABCMeta):
    def __init__(self):
        super().__init__(predict_fct=None)  # predict is identity

        self.nb_directions = int(self.bidirectional) + 1

        self.stacked_lstm = nn.LSTM(self.input_size, self.hidden_size,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout,
                                    bidirectional=self.bidirectional,
                                    batch_first=True)

        self.linear_layer = nn.Linear(self.hidden_size * self.nb_directions * self.nb_output_consider,
                                      self.hidden_FC)
        self.linear_layer_2 = nn.Linear(self.hidden_FC, self.output_size * self.output_time_series_len)

        self.hidden_state_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                       self.input_size,  # repeated later to have batch size
                                                       self.hidden_size),
                                           requires_grad=True)  # parameters are moved to device and learn.
        self.hidden_cell_0 = nn.Parameter(torch.randn(self.num_layers * self.nb_directions,
                                                      self.input_size,  # repeated later to have batch size
                                                      self.hidden_size),
                                          requires_grad=True)  # parameters are moved to device and learn.

    def forward(self, time_series):
        """
        Args:
            time_series: shape L,N, input_dim

        Returns:
        """
        batch_size = 1, time_series.shape[0], 1
        # WIP is backpropagation doing the right job?
        h0, c0 = self.hidden_state_0.repeat(batch_size), self.hidden_cell_0.repeat(batch_size)
        out, _ = self.stacked_lstm(time_series, (h0, c0))  # shape of out is  N,L,Hidden_size * nb_direction
        out = torch.cat((out[:,
                         -self.nb_output_consider:,
                         :self.hidden_size],
                         out[:,
                         :self.nb_output_consider,
                         self.hidden_size:]), 1)
        # this is where the output lies. We take nb_output.. elements. Meaning the h_n, h_n-1...
        # the shape of out at this stage is N,  nb_output_consider, Hidden_size * nb_direction
        out = out.view(-1, self.hidden_size * self.nb_directions * self.nb_output_consider)
        # squeeshing the two last dimensions into one, for input to FC layer.


        out = self.linear_layer(out)
        out = self.activation_fct(out)
        out = self.linear_layer_2(out)
        return out.view(-1, self.output_time_series_len,
                        self.output_size)  # batch size, dim time series output, dim output

    # section ######################################################################
    #  #############################################################################
    # SETTERS GETTERS

    @property
    @abstractmethod
    def input_size(self):
        return self._input_size

    @property
    @abstractmethod
    def output_size(self):
        return self._output_size

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


def factory_parametrised_LSTM(input_size=1, output_size=1,
                              num_layers=1, bidirectional=False,
                              input_time_series_len=1, output_time_series_len=1,
                              nb_output_consider=1,
                              hidden_size=150, dropout=0.,
                              activation_fct=nn.CELU(), hidden_FC=16):
    class Parametrised_LSTM(LSTM):
        def __init__(self):
            self.input_size = input_size
            self.output_size = output_size

            self.input_time_series_len = input_time_series_len
            self.output_time_series_len = output_time_series_len

            self.nb_output_consider = nb_output_consider

            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.hidden_FC = hidden_FC
            super().__init__()
            self.activation_fct = activation_fct  # after init for this reason :
            # https://stackoverflow.com/questions/43080583/attributeerror-cannot-assign-module-before-module-init-call

        # section ######################################################################
        #  #############################################################################
        # SETTERS GETTERS

        @property
        def input_size(self):
            return self._input_size

        @input_size.setter
        def input_size(self, new_input_size):
            if isinstance(new_input_size, int):
                self._input_size = new_input_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def output_size(self):
            return self._output_size

        @output_size.setter
        def output_size(self, new_output_size):
            if isinstance(new_output_size, int):
                self._output_size = new_output_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def hidden_size(self):
            return self._hidden_size

        @hidden_size.setter
        def hidden_size(self, new_hidden_size):
            if isinstance(new_hidden_size, int):
                self._hidden_size = new_hidden_size
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def bidirectional(self):
            return self._bidirectional

        @bidirectional.setter
        def bidirectional(self, new_bidirectional):
            if isinstance(new_bidirectional, bool):
                self._bidirectional = new_bidirectional
            else:
                raise Error_type_setter(f"Argument is not an {str(bool)}.")

        @property
        def num_layers(self):
            return self._num_layers

        @num_layers.setter
        def num_layers(self, new_num_layers):
            if isinstance(new_num_layers, int):
                self._num_layers = new_num_layers
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def dropout(self):
            return self._dropout

        @dropout.setter
        def dropout(self, new_dropout):
            if isinstance(new_dropout, float) and 0 <= new_dropout < 1:
                # : dropout should be a percent between 0 and 1.
                self._dropout = new_dropout
            else:
                raise Error_type_setter(f"Argument is not an {str(float)}.")

        @property
        def input_time_series_len(self):
            return self._time_series_len

        @input_time_series_len.setter
        def input_time_series_len(self, new_input_time_series_len):
            assert new_input_time_series_len > 0, "input_time_series_len should be strictly positive."
            if isinstance(new_input_time_series_len, int):
                self._input_time_series_len = new_input_time_series_len
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def output_time_series_len(self):
            return self._output_time_series_len

        @output_time_series_len.setter
        def output_time_series_len(self, new_output_time_series_len):
            assert new_output_time_series_len > 0, "output_time_series_len should be strictly positive."
            if isinstance(new_output_time_series_len, int):
                self._output_time_series_len = new_output_time_series_len
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def nb_output_consider(self):
            return self._nb_output_consider

        @nb_output_consider.setter
        def nb_output_consider(self, new_nb_output_consider):
            if isinstance(new_nb_output_consider, int):
                self._nb_output_consider = new_nb_output_consider
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

    return Parametrised_LSTM
