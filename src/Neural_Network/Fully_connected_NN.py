# for neural networks
from copy import deepcopy

import torch
import torch.nn as nn

from abc import abstractmethod

# my lib
from priv_lib_error import Error_type_setter
from priv_lib_util.tools import function_iterable


# the class of NN
class Fully_connected_NN(nn.Module):
    # Abstract classes with virtual parameters that are initialized with the function *factory_parametrised_FC_NN*
    #
    # Abstract args:
    # input_size: the size of the input layer.
    # list_hidden_sizes: iterable the input sizes for each hidden layer + output of last hidden layer.
    # output_size: the output size of the neural network.
    # list_biases: list of booleans for specifying which layers use biases.
    # activation_functions: list of activation functions for each layer.
    # dropout: dropout rate for all layers. We do not dropout the first and last layer (input and output layer).
    # predict_function should be a callable function

    @staticmethod
    def my_id(out):
        return out

    # function that from the output returns the prediction. Depends on the problem:
    _predict_fct = my_id  # :default predict_fct. Can be masked with lower child class functions.

    # : the hidden mark "_" is important to pass through the getter.
    def __init__(self):
        """
        Constructor for Neural Network.
        """
        super().__init__()

        # best parameters, keeps track in case of early stopping.
        self.best_weights = None  # init the field best weights.
        self.best_epoch = 0

    def update_best_weights(self, epoch):
        # : We decide to keep a copy instead of saving the model in a file because we might not want to save this model (E.G. if we do a K-FOLD)
        self.best_weights = deepcopy(self.state_dict())
        self.best_epoch = epoch

    # section ######################################################################
    #  #############################################################################
    #  SETTERS / GETTERS
    @property
    @abstractmethod
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, new_input_size):
        if isinstance(new_input_size, int):
            self._input_size = new_input_size
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    @property
    @abstractmethod
    def list_hidden_sizes(self):
        return self._list_hidden_sizes

    @list_hidden_sizes.setter
    def list_hidden_sizes(self, new_list_hidden_sizes):
        if function_iterable.is_iterable(new_list_hidden_sizes):
            self._list_hidden_sizes = new_list_hidden_sizes
        else:
            raise Error_type_setter(f"Argument is not an Iterable.")

    @property
    @abstractmethod
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, new_output_size):
        if isinstance(new_output_size, int):
            self._output_size = new_output_size
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    @property
    @abstractmethod
    def list_biases(self):
        return self._list_biases

    # always set list_biases after list_hidden_sizes:
    @list_biases.setter
    def list_biases(self, new_list_biases):
        if function_iterable.is_iterable(new_list_biases):
            assert len(new_list_biases) == len(self.list_hidden_sizes) + 1
            # :security that the right parameters are given.
            self._list_biases = new_list_biases
        else:
            raise Error_type_setter(f"Argument is not an iterable.")

    @property
    @abstractmethod
    def activation_functions(self):
        return self._activation_functions

    @activation_functions.setter
    def activation_functions(self, new_activation_functions):
        if function_iterable.is_iterable(new_activation_functions):
            self._activation_functions = new_activation_functions
        else:
            raise Error_type_setter(f"Argument is not an iterable.")

    # function that from the output returns the prediction. Depends on the problem:
    @property
    def predict_fct(self):
        return self._predict_fct

    @predict_fct.setter
    def predict_fct(self, new_predict_fct):
        if new_predict_fct is None:
            pass
        else:
            if callable(new_predict_fct):
                self._predict_fct = new_predict_fct
            else:
                raise Error_type_setter(f"Argument is not callable.")

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, new_dropout):
        if isinstance(new_dropout, float) and 0 <= new_dropout < 1:  # : dropout should be a percent between 0 and 1.
            self._dropout = new_dropout
        else:
            raise Error_type_setter(f"Argument is not an {str(float)}.")

    @property
    def best_weights(self):
        return self._best_weights

    @best_weights.setter
    def best_weights(self, new_best_weights):
        self._best_weights = new_best_weights

    # section ######################################################################
    #  #############################################################################
    # rest of methods

    def set_layers(self):  #: mandatory call in the constructor, 
        #: to initialize all the layers and dropout with respect to the parameters created.

        # array of fully connected layers
        self._layers = nn.ModuleList()
        # initialise the input layer
        self._layers.append(nn.Linear(self.input_size, self.list_hidden_sizes[0], self.list_biases[0]))
        # initialise the hidden layers
        for i in range(len(self.list_hidden_sizes) - 1):
            self._layers.append(nn.Linear(self.list_hidden_sizes[i],
                                          self.list_hidden_sizes[i + 1],
                                          self.list_biases[i + 1]))
        # initialise the output layer
        self._layers.append(nn.Linear(self.list_hidden_sizes[-1], self.output_size, self.list_biases[-1]))
        # initialise dropout
        self._apply_dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        # pass through the input layer
        out = self.activation_functions[0](self._layers[0](x))

        # pass through the hidden layers
        for layer_index in range(1, len(self.list_hidden_sizes) - 1):
            out = self.activation_functions[layer_index](self._apply_dropout(self._layers[layer_index](out)))

        # pass through the output layer
        out = self._layers[-1](out)
        return out

    def prediction(self, out):
        """returns the class predicted for each element of the tensor."""
        # gets the class that is max probability
        return self.predict_fct(out)

    def init_weights_of_model(self):
        """Initialise weights of the model such that they have a predefined structure"""
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        """gets called in init_weights_of_model"""
        if type(layer) == nn.Linear and layer.weight.requires_grad and layer.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
            layer.bias.data.fill_(0)


    def save_net(self, path):
        torch.save(self.state_dict(), path)
        return self

    def load_net(self, path):
        self.load_state_dict(torch.load(path))
        return self

# section ######################################################################
#  #############################################################################
# CLASS FACTORY :  creates subclasses of FC NN

def factory_parametrised_FC_NN(input_size, list_hidden_sizes, output_size,
                               list_biases, activation_functions,
                               dropout=0, predict_fct=None):
    class parametrised_FC_NN(Fully_connected_NN):
        def __init__(self):
            super().__init__()
            self.input_size = input_size
            self.list_hidden_sizes = list_hidden_sizes
            self.output_size = output_size
            self.list_biases = list_biases  # should always be defined after list_hidden_sizes.
            self.activation_functions = activation_functions
            self.dropout = dropout
            self.predict_fct = predict_fct

            self.set_layers()  #: mandatory call in the constructor,
            #: to initialize all the layers and dropout with respect to the parameters created.

    return parametrised_FC_NN
