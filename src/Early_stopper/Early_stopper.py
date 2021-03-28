from abc import abstractmethod

import numpy as np

import torch
import torch.utils.data


# todo the path things, saving the NN.


class Early_stopper:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    References:
            https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    In order to use it, one initializes it and then call it with the corresponding data.
    The call returns whether or not we should "early stop" training.
    """

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.highest_loss = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, neural_network, losses, epoch):
        current_loss = losses[epoch]

        if self.is_early_stop(current_loss, epoch):
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.highest_loss = current_loss
            self.save_checkpoint(current_loss, neural_network)
            self.counter = 0

        return False

    @abstractmethod
    def is_early_stop(self, losses, epoch):
        pass

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss