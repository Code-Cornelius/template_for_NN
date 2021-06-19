from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from priv_lib_error import Error_type_setter

from src.nn_classes.architecture.savable_net import Savable_net

def factory_parametrised_RNN(input_dim=1, output_dim=1, num_layers=1, bidirectional=False, input_time_series_len=1,
                             output_time_series_len=1, nb_output_consider=1, hidden_size=150, dropout=0.,
                             activation_fct=nn.CELU(), hidden_FC=64, * , rnn_class, Parent):
    """

    Args:
        input_dim:
        output_dim:
        num_layers:
        bidirectional:
        input_time_series_len:
        output_time_series_len:
        nb_output_consider:
        hidden_size:
        dropout:
        activation_fct:
        hidden_FC:
        rnn_class: module either RNN OR LSTM
        Parent:  GRU OR LSTM the special classes.

    Returns:

    """

    class Parametrised_RNN(Parent):
        def __init__(self):
            self.input_dim = input_dim
            self.output_dim = output_dim

            self.input_time_series_len = input_time_series_len
            self.output_time_series_len = output_time_series_len

            self.nb_output_consider = nb_output_consider

            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.hidden_FC = hidden_FC
            self.rnn_class = rnn_class
            super().__init__()
            self.activation_fct = activation_fct  # after init for this reason :
            # https://stackoverflow.com/questions/43080583/attributeerror-cannot-assign-module-before-module-init-call

        # section ######################################################################
        #  #############################################################################
        # SETTERS GETTERS

        @property
        def input_dim(self):
            return self._input_dim

        @input_dim.setter
        def input_dim(self, new_input_dim):
            if isinstance(new_input_dim, int):
                self._input_dim = new_input_dim
            else:
                raise Error_type_setter(f"Argument is not an {str(int)}.")

        @property
        def output_dim(self):
            return self._output_dim

        @output_dim.setter
        def output_dim(self, new_output_dim):
            if isinstance(new_output_dim, int):
                self._output_dim = new_output_dim
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
                if isinstance(new_dropout, int) and not (new_dropout):  # dropout == 0
                    self._dropout = float(new_dropout)
                else:
                    raise Error_type_setter(f"Argument is not an {str(float)}.")

        @property
        def input_time_series_len(self):
            return self._input_time_series_len

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

        @property
        def rnn_class(self):
            return self._rnn_class

        @rnn_class.setter
        def rnn_class(self, new_nn_class):
            self._rnn_class = new_nn_class

    return Parametrised_RNN
