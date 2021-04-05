import sklearn.metrics
import torch.utils
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.Neural_Network.NN_fcts import are_at_least_one_None, raise_if_not_all_None, device, nn_predict
from src.Training_stopper.Early_stopper_vanilla import Early_stopper_vanilla

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
           X_train_on_device, Y_train_on_device, Y_train,
           params_training,
           training_losses, training_accuracy,
           early_stopper_validation=Early_stopper_vanilla(),
           early_stopper_training=Early_stopper_vanilla(),
           X_val_on_device=None, Y_val_on_device=None, Y_val=None,
           validation_losses=None, validation_accuracy=None,
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
    list_params_validat = [X_val_on_device, Y_val_on_device,
                              Y_val, validation_losses,
                              validation_accuracy]

    is_validat_included = not are_at_least_one_None(list_params_validat)  #: equivalent to are all not None ?
    if not is_validat_included:
        raise_if_not_all_None(list_params_validat)
    else : #: if we need validation
        # create data validat_loader : load validation data in batches
        validat_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_on_device, Y_val_on_device),
                                                     batch_size=params_training.batch_size, shuffle=True,
                                                     num_workers=0)  # num_workers can be increased, only under Linux.

    # create data train_loader : load training data in batches
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_on_device, Y_train_on_device),
                                         batch_size=params_training.batch_size, shuffle=True,
                                         num_workers=0)  # num_workers can be increased, only under Linux.


    # pick loss function and optimizer
    criterion = params_training.criterion
    optimiser = params_training.optimiser(net.parameters(), **params_training.dict_params_optimiser)

    for epoch in tqdm(range(params_training.epochs), disable=silent):  # disable unable the print.
        ###################
        # train the model #
        ###################
        train_loss = 0
        for i, (batch_X, batch_y) in enumerate(train_loader, 0):
            # closure needed for some algorithm.

            # get batch
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            def closure():
                # set gradients to zero
                optimiser.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                # Do forward and backward pass
                loss = criterion(net(batch_X), batch_y)  #: compute the loss : difference of result and expectation
                loss.backward()  # : compute the gradients
                return loss

            # Optimisation step
            optimiser.step(closure=closure)  # : update the weights

            # you need to call again criterion, as we cannot store the criterion result:
            train_loss += criterion(net(batch_X), batch_y).item() * batch_X.shape[0]
            #: weight the loss accordingly. That is the reason why using average is flawed.

        _update_history(X_train_on_device, X_val_on_device, Y_train, Y_train_on_device, Y_val, Y_val_on_device,
                        compute_accuracy, criterion, epoch, is_validat_included, net, train_loss, training_accuracy,
                        training_losses, validation_accuracy, validation_losses)

        ######################
        #   Early Stopping   #
        ######################
        # Check if NN has not improved with respect to one of the two criteria.
        # If has not, we do not improve the best_weights of the NN
        if early_stopper_validation.has_improved_last_epoch and early_stopper_training.has_improved_last_epoch:
            net.update_best_weights(epoch)

        # Calculations to see if it's time to stop early:
        if is_validat_included:
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
    return _return_the_stop(net, epoch, early_stopper_validation, early_stopper_training)


def _return_the_stop(net, current_epoch, *args):  # args should be early_stoppers (or none if not defined)
    # multiple early_stoppers can't break at the same time, because there will be a first that breaks out the loop first.
    # if no early_stopper broke, return the current epoch.
    for stopper in args:
        if stopper.is_stopped():  #: check if the stopper is none or actually of type early stop.
            (net.load_state_dict(net.best_weights))  # .to(device)
            return net.best_epoch
    return current_epoch


def _update_history(X_train_on_device, X_val_on_device, Y_train, Y_train_on_device, Y_val, Y_val_on_device,
                    compute_accuracy, criterion, epoch, is_validat_included, net, train_loss, train_accuracy,
                    training_losses, validat_accuracy, validat_losses):
    ######################
    #   Training Losses  #
    ######################
    # Normalize and save the loss over the current epoch:
    training_losses[epoch] = train_loss / (Y_train_on_device.shape[0])

    ######################
    # Training Accuracy  #
    ######################
    if compute_accuracy:
        train_accuracy[epoch] = sklearn.metrics.accuracy_score(
            nn_predict(net, X_train_on_device),
            Y_train.reshape(-1, 1)
        )
        # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
        # :sklearn can't access data on gpu.

    ######################
    #   Validation Loss  #
    ######################
    if is_validat_included:
        validat_losses[epoch] = criterion(net(X_val_on_device),
                                          Y_val_on_device).item()
        #######################
        # Validation Accuracy #
        #######################
        if compute_accuracy:
            validat_accuracy[epoch] = sklearn.metrics.accuracy_score(
                nn_predict(net, X_val_on_device),
                Y_val.reshape(-1, 1)
            )
        # :here the reshape is assuming we use Cross Entropy and the data outside is set in the right format.
        # :sklearn can't access data on gpu.
