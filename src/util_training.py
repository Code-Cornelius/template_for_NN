import functools

import numpy as np
import torch
import torch.cuda

def decorator_train_disable_no_grad(func):
    """
    Be careful with it, if you wrap something already wrapped, the wrapping will disapear !
    Args:
        func:

    Returns:

    """

    @functools.wraps(func)
    def wrapper_decorator_train_disable_no_grad(net, *args, **kwargs):
        net.train(mode=False)  # Disable dropout and normalisation
        with torch.no_grad():  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
            ans = func(net, *args, **kwargs)
        net.train(mode=True)  # Re-able dropout and normalisation
        return ans

    return wrapper_decorator_train_disable_no_grad


def pytorch_device_setting(type=''):
    """
    Semantics : sets the device for NeuralNetwork computations.
    Put nothing for automatic choice.
    If cpu given, sets cpu
    else, see if cuda available, otherwise cpu.

    Args:
        type:

    Returns:

    """
    device = torch.device('cpu') if type == 'cpu' else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Script running on device : ", device)
    return device


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
