from src.Neural_Network.NN_fcts import decorator_train_disable_no_grad


@decorator_train_disable_no_grad
def nn_predict(net, data_to_predict):
    """
    Semantics : pass data_to_predict through the neural network and returns its prediction.
    The output data is going through the net.prediction() function.
    Also, we request the device, where the input, the net, and output lies.

    Condition: net has the method prediction.

    Args:
        net:
        data_to_predict:

    Returns:

    """
    #~~~~~~~~~~~~~~~~~~ to device for optimal speed, though we take the data back with .cpu().
    # we do not put the data on GPU! As the overhead might be too much.
    data_predicted = net.prediction(net(data_to_predict))  # forward pass
    return data_predicted


def nn_predict_ans2cpu(net, data_to_predict):
    return nn_predict(net, data_to_predict).cpu()