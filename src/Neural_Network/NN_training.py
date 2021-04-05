import torch.utils.data
import sklearn.model_selection
from src.Neural_Network.NN_fcts import device, are_at_least_one_None, raise_if_not_all_None, nn_predict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def nn_fit(net,
           X_train_on_device, Y_train_on_device,
           Y_train,
           params_training,
           training_losses, training_accuracy,
           X_val_on_device=None, Y_val_on_device=None, Y_val=None,
           validation_losses=None, validation_accuracy=None,
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
        early_stopper_training:
        early_stopper_validation:
        compute_accuracy: if True, training_accuracy and validation_accuracy are not updated.
        silent: verbose.

    Returns: return epoch of best net and updates the value passed in
    training_losses, training_accuracy,
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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            def closure():
                # set gradients to zero
                optimiser.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                # Do forward and backward pass
                loss = criterion(net(batch_X), batch_y)  # compute the loss : difference of result and expectation
                loss.backward()  # compute the gradients
                return loss

            # Optimisation step
            optimiser.step(closure=closure)  # update the weights

            # you need to call again criterion, as we cannot store the criterion result.
            train_loss += criterion(net(batch_X), batch_y).item() * batch_X.shape[0]  # weight the loss accordingly

        update_history(X_train_on_device, X_val_on_device, Y_train,
                       Y_train_on_device, Y_val, Y_val_on_device,
                       compute_accuracy, criterion, epoch, is_validation_included,
                       net, train_loss, training_accuracy,
                       training_losses, validation_accuracy, validation_losses)

        # Check if NN has not improved with respect to one of the two critereas. If has not, we do not improve the best_weights of the NN
        if early_stopper_validation is not None and early_stopper_training is not None:
            if early_stopper_validation.has_improved_last_epoch and early_stopper_training.has_improved_last_epoch:
                net.update_best_weights(epoch)

        if early_stopper_training is None and early_stopper_validation is not None and early_stopper_validation.has_improved_last_epoch:
            net.update_best_weights(epoch)

        if early_stopper_validation is None and early_stopper_training is not None and early_stopper_training.has_improved_last_epoch:
            net.update_best_weights(epoch)

        # Calculations to see if it's time to stop early:
        if is_validation_included and early_stopper_validation is not None:
            if early_stopper_validation(net, validation_losses, epoch):
                if not silent: print("Terminated epochs, with early stopper validation at epoch {}.".format(epoch))
                break  #: get out of epochs
        if early_stopper_training is not None:
            if early_stopper_training(net, training_losses, epoch):
                if not silent: print("Terminated epochs, with early stopper training at epoch {}.".format(epoch))
                break  #: get out of epochs.

        if PLOT_WHILE_TRAIN:
            if epoch % FREQ_NEW_IMAGE == 0:
                plot_while_training(params_training, training_losses, validation_losses)

    # ~~~~~~~~ end of the for in epoch. Training
    return return_the_stop(net, epoch, early_stopper_validation, early_stopper_training)


def return_the_stop(net, current_epoch, *args):  # args should be early_stoppers (or none if not defined)
    # multiple early_stoppers can't break at the same time, because there will be a first that breaks out the loop first.
    # if no early_stopper broke, return the current epoch.
    for stopper in args:
        if stopper is not None and stopper.is_stopped():  #: check if the stopper is none or actually of type early stop.
            (net.load_state_dict(net.best_weights))  # .to(device)
            return net.best_epoch
    return current_epoch


def update_history(X_train_on_device, X_val_on_device, Y_train, Y_train_on_device, Y_val, Y_val_on_device,
                   compute_accuracy, criterion, epoch, is_validation_included, net, train_loss, training_accuracy,
                   training_losses, validation_accuracy, validation_losses):
    # Normalize and save the loss over the current epoch:
    training_losses[epoch] = train_loss / (Y_train_on_device.shape[0])
    if compute_accuracy:
        training_accuracy[epoch] = sklearn.metrics.accuracy_score(
            nn_predict(net, X_train_on_device),
            Y_train.reshape(-1, 1)
        )
        # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
        # :sklearn can't access data on gpu.
    # Calculate validation loss for the current epoch
    if is_validation_included:
        validation_losses[epoch] = criterion(net(X_val_on_device),
                                             Y_val_on_device).item()
        if compute_accuracy:
            validation_accuracy[epoch] = sklearn.metrics.accuracy_score(
                nn_predict(net, X_val_on_device),
                Y_val.reshape(-1, 1)
            )
        # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
        # :sklearn can't access data on gpu.


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

    Returns: the data with history of training and best epoch for training. Net is modified in the progress.
    If validation is given:
        returns trained/validation accuracy, trained/validation loss;
    Else:
        returns trained accuracy then loss;
    Whenever the accuracy is not requested, the accuracy vector is zero.
    """
    # Prepare Training set
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train = data_Y[indic_train_Y]  # : useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    # prepare for iteration over epochs:
    training_losses = np.full(params_training.epochs, np.nan)
    training_accuracy = np.full(params_training.epochs, np.nan)

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
        validation_losses = np.full(params_training.epochs, np.nan)
        validation_accuracy = np.full(params_training.epochs, np.nan)

        # essentially, we need to check what is the max epoch:
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses,
                                training_accuracy,
                                X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, Y_val=Y_val,
                                validation_losses=validation_losses, validation_accuracy=validation_accuracy,
                                early_stopper_training=early_stopper_training,
                                early_stopper_validation=early_stopper_validation,
                                compute_accuracy=compute_accuracy, silent=silent)

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
        epoch_best_net = nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses,
                                training_accuracy,
                                early_stopper_training=early_stopper_training,
                                early_stopper_validation=early_stopper_validation,
                                compute_accuracy=compute_accuracy, silent=silent)
        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, training_losses,
                    epoch_best_net)
        else:
            return (training_accuracy, training_losses,
                    epoch_best_net)
