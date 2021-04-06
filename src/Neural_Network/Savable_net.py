# for neural networks
from copy import deepcopy

import torch
import torch.nn as nn

from abc import abstractmethod

# my lib
from priv_lib_error import Error_type_setter
from priv_lib_util.tools import function_iterable


class Savable_net(nn.Module):
    """
    Args:
        best_weights:
        best_epoch:
    Class Args:
        _predict_fct
    """

    @staticmethod
    def my_id(out):
        return out

    # function that from the output returns the prediction. Depends on the problem:
    _predict_fct = my_id

    # :default predict_fct. Can be masked with lower child class functions.
    # : the hidden mark "_" is important to not pass through the setter.
    # we set the class variable, that is also defined as an object variable unless redefined!

    def __init__(self, *args, **kwargs):
        """
        Constructor for Neural Network.
        """
        super().__init__()
        # best parameters, keeps track in case of early stopping.
        self.best_weights = None  # init the field best weights.
        self.best_epoch = 0

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

    def update_best_weights(self, epoch):
        # : We decide to keep a copy instead of saving the model in a file because we might not want to save this model (E.G. if we do a K-FOLD)
        self.best_weights = deepcopy(self.state_dict())
        self.best_epoch = epoch

    # section ######################################################################
    #  #############################################################################
    # SETTER GETTER

    @property
    def predict_fct(self):
        # function that from the output returns the prediction. Depends on the problem:
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
    def best_weights(self):
        return self._best_weights

    @best_weights.setter
    def best_weights(self, new_best_weights):
        self._best_weights = new_best_weights
