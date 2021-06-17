from abc import abstractmethod, ABCMeta

import torch.nn as nn
# my lib
from priv_lib_error import Error_type_setter
from priv_lib_util.tools import function_iterable

# Savable_net
from src.nn_classes.architecture.savable_net import Savable_net


class residual_split(nn.Module):
    def __init__(self, in_p, out_p):
        super().__init__()
        self.first_layer = nn.Linear(in_p, out_p, bias=True)
        self.second_layer = nn.Linear(in_p, out_p, bias=True)

    def forward(self, x):
        first = self.first_layer(x)
        second = self.second_layer(x)
        return first + second


class Free_NN(Savable_net, metaclass=ABCMeta):
    def __init__(self, predict_fct, *args, **kwargs):
        """
        Constructor for Neural Network.
        """
        super().__init__(predict_fct, *args, **kwargs)

    # section ######################################################################
    #  #############################################################################
    #  SETTERS / GETTERS

    @property
    @abstractmethod  # ABSTRACT FIELD
    def layers_instructions(self):
        return self._layers_instructions

    @layers_instructions.setter
    def layers_instructions(self, new_layers_instructions):
        if function_iterable.is_iterable(new_layers_instructions):
            self._layers_instructions = new_layers_instructions
        else:
            raise Error_type_setter(f"Argument is not an iterable.")

    # section ######################################################################
    #  #############################################################################
    # rest of methods

    def set_layers(self):  #: mandatory call in the constructor,
        #: to initialize all the layers and dropout with respect to the parameters created.

        # array with the layers
        self._layers = nn.ModuleList()

        for module in self.layers_instructions:
            self._layers.append(module)

    def forward(self, x):
        for i in range(len(self._layers)):
            x = self._layers[i](x)
        return x


# section ######################################################################
#  #############################################################################
# CLASS FACTORY :  creates subclasses of Free NN

def factory_parametrised_Free_NN(param_layers_instructions, param_predict_fct=None):
    class Parametrised_Free_NN(Free_NN):
        # defining attributes this way shadows the abstract properties from parents.
        layers_instructions = param_layers_instructions

        def __init__(self):
            super().__init__(predict_fct=param_predict_fct)
            self.set_layers()  #: mandatory call in the constructor,
            # :to initialize all the layers and dropout with respect to the parameters created.

    return Parametrised_Free_NN


# channels is the depth of input.

input_size = 100
list = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
        nn.BatchNorm1d(input_size - 4),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
        residual_split(input_size - 8, 300),
        nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
        nn.Linear(300 // 4, 50, bias=True),
        nn.Linear(50, 15, bias=True),
        nn.Linear(15, 2, bias=True)
        ]
CNN = factory_parametrised_Free_NN(list)

CNN()
