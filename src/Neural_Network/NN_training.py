# parameters

# from useful_functions import *

# for neural networks
import torch.utils.data

import sklearn.model_selection

from src.Neural_Network.NN_fcts import device, are_at_least_one_None, raise_if_not_all_None, nn_predict
from src.kfold import *
from tqdm import tqdm

PLOT_WHILE_TRAIN = False
if PLOT_WHILE_TRAIN:
    fig = plt.figure()
    ax = fig.add_subplot(111)

FREQ_NEW_IMAGE = 40


def plot_while_training(params_training, training_losses, validation_losses):
    ax.clear()
    plt.semilogy(range(params_training.epochs), training_losses, 'b', label='Train Loss')
    plt.semilogy(range(params_training.epochs), validation_losses, 'r', label='Validation Loss')
    plt.legend(loc="best")
    plt.pause(0.0001)


def nn_fit(net, X_train_on_device, Y_train_on_device, Y_train,
           params_training,
           training_losses, training_accuracy,
           X_val_on_device=None, Y_val_on_device=None, Y_val=None,
           validation_losses=None, validation_accuracy=None,
           max_through_epoch=None,
           early_stopper_training=None, early_stopper_validation=None,
           compute_accuracy=False,
           silent=False):
    """

    Args:
        net: model
        X_train_on_device:
        Y_train_on_device:
        Y_train:
        params_training:
        training_losses: always passed, numpy array where values are stored.
        training_accuracy: always passed, numpy array where values are stored.
        Passed, even though compute_accuracy is false.
        X_val_on_device:
        Y_val_on_device:
        Y_val:
        validation_losses: always passed, numpy array where values are stored.
        validation_accuracy: always passed, numpy array where values are stored.
        Passed, even though compute_accuracy is false.
        max_through_epoch:
        early_stopper_training:
        early_stopper_validation:
        compute_accuracy: if True, training_accuracy and validation_accuracy are not updated.
        silent: verbose.

    Returns: nothing but updates the value passed, training_losses, training_accuracy,
    validation_losses, validation_accuracy, max_through_epoch.

    """
    # condition if we use validation set.
    list_params_validation = [X_val_on_device, Y_val_on_device,
                              Y_val, validation_losses,
                              validation_accuracy]

    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to are all not None ?
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # create data loader
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_on_device, Y_train_on_device),
                                         batch_size=params_training.batch_size,
                                         shuffle=True,
                                         num_workers=0)  # num_workers can be increased, only under Linux.

    # pick loss function and optimizer
    criterion = params_training.criterion
    optimiser = params_training.optimiser(net.parameters(),
                                          **params_training.dict_params_optimiser)

    for epoch in tqdm(range(params_training.epochs), disable=silent):  # disable unable the print.
        train_loss = 0
        for i, (batch_X, batch_y) in enumerate(loader, 0):
            # closure needed for some algorithm.

            # get batch
            # squeeze batch y in order to have the right format. not the good size for the results
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # WIP THERE WAS A SQUEEZE HERE ON Y

            def closure():
                # set gradients to zero
                optimiser.zero_grad()

                # Do forward and backward pass
                loss = criterion(net(batch_X), batch_y)
                loss.backward()
                return loss

            # Optimisation step
            optimiser.step(closure=closure)

            # you need to call again criterion unless you do not need the closure.
            train_loss += criterion(net(batch_X), batch_y).item() * batch_X.shape[0]  # weight the loss accordingly

        # Normalize and save the loss over the current epoch:
        training_losses[epoch] = train_loss / (Y_train_on_device.shape[0])
        if compute_accuracy:
            training_accuracy[epoch] = sklearn.metrics.accuracy_score(
                nn_predict(net, X_train_on_device), Y_train.reshape(-1,1))
            # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
            # :sklearn can't access data on gpu.

        # Calculate validation loss for the current epoch
        if is_validation_included:
            validation_losses[epoch] = criterion(net(X_val_on_device),
                                                 Y_val_on_device).item()  # WIP THERE WAS A SQUEEZE HERE ON Y
            if compute_accuracy:
                validation_accuracy[epoch] = sklearn.metrics.accuracy_score(
                    nn_predict(net, X_val_on_device), Y_val.reshape(                    -1, 1))
            # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
                # :sklearn can't access data on gpu.

            # Calculations to see if it's time to stop early:
            if early_stopper_validation is not None:
                if early_stopper_validation(net, validation_losses, epoch):
                    break  #: get out of epochs
        if early_stopper_training is not None:
            if early_stopper_training(net, training_losses, epoch):
                break  #: get out of epochs.

        if PLOT_WHILE_TRAIN:
            if epoch % FREQ_NEW_IMAGE == 0:
                plot_while_training(params_training, training_losses, validation_losses)

    # ~~~~~~~~ end of the for in epoch.
    # we change the value of max_through_epoch:
    max_through_epoch[0] = epoch + 1  #: +1 because it starts at zero so the real value is shifted.


def nn_train(net, data_X, data_Y,
             params_training,
             indic_train_X, indic_train_Y,
             indic_validation_X=None, indic_validation_Y=None,
             early_stopper_training=None, early_stopper_validation=None,
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

    Returns: Trained net and the data.
    If validation is given:
        returns trained/validation accuracy, trained/validation loss;
    Else:
        returns trained accuracy then loss;
    Whenever the accuracy is not requested, the accuracy vector is zero.
    """
    max_through_epoch = [0]  # : nb of epochs that the NN has back propagated over.
    #: we need to use a container because 0 is immutable, and we want that value to change inside of fit.

    # Prepare Training set
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train = data_Y[indic_train_Y]  # : useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    # prepare for iteration over epochs:
    training_losses = np.full(params_training.epochs, np.nan)
    training_accuracy = np.full(params_training.epochs, np.nan)

    # condition if we use validation set:
    list_params_validation = [indic_validation_X, indic_validation_Y]
    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to are all not None ?
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indic_validation_X].to(device)
        Y_val = data_Y[indic_validation_Y]  # :useful for using it in order to compute accuracy.
        Y_val_on_device = Y_val.to(device)
        validation_losses = np.full(params_training.epochs, np.nan)
        validation_accuracy = np.full(params_training.epochs, np.nan)

        # essentially, we need to check what is the max epoch:
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses, training_accuracy,
               X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, Y_val=Y_val,
               validation_losses=validation_losses, validation_accuracy=validation_accuracy,
               max_through_epoch=max_through_epoch, early_stopper_training=early_stopper_training,
               early_stopper_validation=early_stopper_validation, compute_accuracy=compute_accuracy, silent=silent)

        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch[0])
        else:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch[0])

    # if no validation set
    else:
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses, training_accuracy,
               max_through_epoch=max_through_epoch, early_stopper_training=early_stopper_training,
               early_stopper_validation=early_stopper_validation, compute_accuracy=compute_accuracy, silent=silent)
        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, training_losses,
                    max_through_epoch[0])
        else:
            return (training_accuracy, training_losses,
                    max_through_epoch[0])


