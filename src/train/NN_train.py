from src.Neural_Network.NN_fcts import are_at_least_one_None, raise_if_not_all_None
import numpy as np

from src.train.NN_fit import nn_fit
from src.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla


def nn_train(net, data_X, data_Y,
             params_training,
             indic_train_X, indic_train_Y,
             early_stopper_validation=Early_stopper_vanilla(), early_stopper_training=Early_stopper_vanilla(),
             indic_validation_X=None, indic_validation_Y=None,
             compute_accuracy=False,
             silent=False):
    """
    Semantics : Given the net, we train it upon data.
    For optimisation reasons, we pass the indices.
    Args:
        net:
        data_X:  tensor
        data_Y:  tensor
        params_training:
        indic_train_X:
        indic_train_Y:
        indic_validation_X:
        indic_validation_Y:
        early_stopper_training: early_stopper_training type,
        early_stopper_validation: early_stopper_validation type,
        compute_accuracy: no impact here, impacts nn_fit.
        silent:

    Returns: the data with history of training and best epoch for training. Net is modified in the progress.
    If validation is given:
        returns trained/validation accuracy, trained/validation loss;
    Else:
        returns trained accuracy then loss;
    Whenever the accuracy is not requested, the accuracy vector is zero.
    """
    # Prepare Training set
    device = params_training.device
    epoch = params_training.epochs
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train = data_Y[indic_train_Y]  # : useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    # prepare for iteration over epochs:
    training_losses = np.full(epoch, np.nan)
    training_accuracy = np.full(epoch, np.nan)

    # condition if we use validation set:
    list_params_validation = [indic_validation_X, indic_validation_Y]
    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to "are all not None ?"
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indic_validation_X].to(device)
        Y_val = data_Y[indic_validation_Y]  # :useful for using it in order to compute accuracy.
        Y_val_on_device = Y_val.to(device)
        validation_losses = np.full(epoch, np.nan)
        validation_accuracy = np.full(epoch, np.nan)

        # essentially, we need to check what is the max epoch:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, Y_train,
                                params_training,
                                training_losses,
                                training_accuracy,
                                early_stopper_validation=early_stopper_validation,
                                early_stopper_training=early_stopper_training,
                                X_val_on_device=X_val_on_device,
                                Y_val_on_device=Y_val_on_device, Y_val=Y_val,
                                validation_losses=validation_losses,
                                validation_accuracy=validation_accuracy,
                                compute_accuracy=compute_accuracy,
                                silent=silent)

        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    epoch_best_net)
        else:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    epoch_best_net)

    # if no validation set
    else:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, Y_train,
                                params_training,
                                training_losses, training_accuracy,
                                early_stopper_validation=early_stopper_validation,
                                early_stopper_training=early_stopper_training,
                                compute_accuracy=compute_accuracy,
                                silent=silent)
        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, training_losses,
                    epoch_best_net)
        else:
            return (training_accuracy, training_losses,
                    epoch_best_net)
