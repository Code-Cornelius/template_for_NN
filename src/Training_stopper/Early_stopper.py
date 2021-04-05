from copy import deepcopy
from abc import abstractmethod

import numpy as np

import torch
import torch.utils.data

# todo the path things, saving the NN.
DEBUG = False


class Early_stopper(object):
    """
    ABSTRACT CLASS

    Early stops the training if validation loss doesn't improve after a given patience.
    References:
            https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    In order to use it, one initializes it and then call it with the corresponding data.
    The call returns whether or not we should "early stop" training.


    The behavior is you set an early_stopper that you call whenever to test condition.
    In order to extend the class, one shall define the abstract method is_early_stop.
    """

    def __init__(self, patience=50, silent=True, delta=0.1, print_func=print):
        #TODO REFACTOR NEURALNETWORK INTO NET
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            silent (bool): If False, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. In percent.
                            Default: 0.1
            print_func (function): trace print function.
                            Default: print
        """
        self._patience = patience
        self._silent = silent
        self._counter = 0
        self._lowest_loss = np.Inf
        self._delta = delta
        self._print_func = print_func

        # for retrieving the best result
        self._early_stopped = False
        self.has_improved_last_epoch = True

    def __call__(self, neural_network, losses, epoch):
        if self._is_early_stop(losses, epoch):
            self._counter += 1
            self.has_improved_last_epoch = False  # : flag giving information about the performance of the NN
            if DEBUG:
                self._print_func(f'EarlyStopping counter: {self._counter} out of {self._patience}')

            #early stop triggered
            if self._counter >= self._patience:
                self._early_stopped = True
                return True
        else:
            self.has_improved_last_epoch = True # : flag giving information about the performance of the NN
            self._lowest_loss = min(losses[epoch], self._lowest_loss)
            self._counter = 0
        return False

    @abstractmethod
    def _is_early_stop(self, losses, epoch):
        """
        should be a const method
        The requirements are:

        Args:
            losses:
            epoch:

        Returns: boolean answering the question should we stop early.

        """
        pass

    def is_stopped(self):
        return self._early_stopped

    def reset(self):
        """ Allows to reset the log of the early stopper between kfold for example."""
        self._early_stopped = False  # if we train again, then we reset early_stopped.
        self.has_improved_last_epoch = True
        self._lowest_loss = np.Inf
        self._counter = 0

