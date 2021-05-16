from src.nn_classes.architecture.nn_fcts import are_at_least_one_None, raise_if_not_all_None
import numpy as np

from src.train.NN_fit import nn_fit
from src.nn_classes.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla


def nn_train(net, data_X, data_Y,
             params_training,
             indic_train_X, indic_train_Y,
             early_stoppers=(Early_stopper_vanilla(),),
             indic_validation_X=None, indic_validation_Y=None,
             silent=False):
    """
    Semantics : Given the net, we train it upon data.
    For optimisation reasons, we pass the indices.
    Args:
        net:
        data_X: tensor
        data_Y: tensor
        params_training: NNTrainParameters. parameters used for training
        indic_train_X: indices of values from data_X to be used for training
        indic_train_Y: indices of values from data_Y to be used for training
        early_stoppers: iterable of Early_stopper. Used for deciding if the training should stop early.
            Preferably immutable to insure no changes.
        indic_validation_X: indices of values from data_X to be used for validation, None if validation is not performed
        indic_validation_Y: indices of values from data_Y to be used for validation, None if validation is not performed
        silent: verbose.

    Returns: the data with history of training and all other metrics
             and best epoch for training. Net is modified in the progress.

    Post-condition :
        early_stoppers not changed.
    """

    # Prepare Training set
    device = params_training.device
    epoch = params_training.epochs
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train_on_device = data_Y[indic_train_Y].to(device)

    # initialise the training history for loss and any other metric included
    history = {'training': {}}
    history['training']['loss'] = np.full(epoch, np.nan)

    for metric in params_training.metrics:
        history['training'][metric.name] = np.full(epoch, np.nan)

    # condition if we use validation set:
    list_params_validation = [indic_validation_X, indic_validation_Y]
    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to "are all not None ?"
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indic_validation_X].to(device)
        Y_val_on_device = data_Y[indic_validation_Y].to(device)

        # initialise the validation history for loss and any other metrics included
        # initialise with nans such that no plot if no value.
        history['validation'] = {}
        history['validation']['loss'] = np.full(epoch, np.nan)

        for metric in params_training.metrics:
            # initialise with nans such that no plot if no value.
            history['validation'][metric.name] = np.full(epoch, np.nan)

        # essentially, we need to check what is the max epoch:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, params_training, history, early_stoppers,
                                X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, silent=silent)

    # if no validation set
    else:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, params_training, history, early_stoppers,
                                silent=silent)

    return history, epoch_best_net
