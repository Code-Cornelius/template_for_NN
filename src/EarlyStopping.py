from abc import abstractmethod

import numpy as np

import torch
import torch.utils.data


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    References:
            https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
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
        # todo make it work without giving validation losses. The function is_early_stop has to be adjusted as well
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
    def is_early_stop(self, loss, epoch):
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


class EarlyStopperTraining(EarlyStopping):

    def __init__(self):
        super().__init__()

    def is_early_stop(self, training_loss, epoch):
        # if the percentage of change of the actual loss wrt to any loss for the 20 previous loss is less than 10%, then stop.
        cdt2 = epoch > 20 and all(
            difference_percentage < 0.1 for difference_percentage in
            [abs(previous_loss - training_loss[epoch]) / previous_loss for previous_loss in
             training_loss[epoch - 20:epoch]
             ]
        )
        return cdt2


class EarlyStoppingValidation(EarlyStopping):
    def __init__(self):
        super().__init__()
        self.highest_loss = None

    def is_early_stop(self, val_loss, epoch):
        if self.highest_loss is None:
            self.highest_loss = val_loss[epoch]
            # self.save_checkpoint(val_loss[epoch], neural_network)
            return False
        return self.highest_loss + self.delta < val_loss[epoch]
