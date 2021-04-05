import functools

import torch
import torch.cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def are_at_least_one_None(list_parameters):
    """returns list_parameters.at least one.is_None"""
    for parameter in list_parameters:
        if parameter is None:
            return True
        else:
            continue
    return False


def raise_if_not_all_None(list_parameters):
    """ if one is not None, throws an error"""
    for parameter in list_parameters:
        if not parameter is None:
            raise ValueError("Given a parameter not None while the others are. "
                             "Is it a mistake ? Parameter not None : " + str(parameter))
    return


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

@decorator_train_disable_no_grad
def nn_predict(net, data_to_predict):
    """
    Semantics : pass data_to_predict through the neural network and returns its prediction.
    # do a single predictive forward pass on net (takes & returns numpy arrays)

    Condition: net has the method prediction.

    Args:
        net:
        data_to_predict:

    Returns:

    """
    # to device for optimal speed, though we take the data back with .cpu().
    data_predicted = net.prediction(net(data_to_predict.to(device))).cpu()  # forward pass
    return data_predicted


def pytorch_device_setting(type="cpu"):
    """
    Semantics : sets the device for NeuralNetwork computations.
    Put nothing for automatic choice.
    Args:
        type:

    Returns:

    """
    device = torch.device("cpu") if type == "cpu" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


