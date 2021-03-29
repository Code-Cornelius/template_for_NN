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
            raise ValueError(str(parameter))  # wip is it raising the way it should?
    return


def nn_predict(net, data_to_predict):
    """
    Semantics : pass data_to_predict through the neural network and returns its prediction.

    Condition: net has the method prediction.

    Args:
        net:
        data_to_predict:

    Returns:

    """
    # do a single predictive forward pass on net (takes & returns numpy arrays)
    net.train(mode=False)  # Disable dropout

    # to device for optimal speed, though we take the data back with .cpu().
    data_predicted = net.prediction(net(data_to_predict.to(device))).detach().cpu()  # forward pass

    net.train(mode=True)  # Re-able dropout
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