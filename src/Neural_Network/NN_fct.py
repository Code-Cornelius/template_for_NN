# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import seaborn as sns  # for the display
import pandas as pd  # for dataframes
import time  # computational time
import scipy.stats as si

# from useful_functions import *

# for neural networks
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import sklearn.model_selection

from src.Neural_Network.Fully_connected_NN import *
from src.Neural_Network.NNTrainParameters import *
from src.Neural_Network.NN_plots import *
from src.kfold import *
from src.Early_stopper import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_None(parameter):
    return parameter is None


def are_at_least_one_None(list_parameters):
    """returns list_parameters.at least one.is_None"""
    for parameter in list_parameters:
        if is_None(parameter):
            return True
        else:
            continue
    return False


def raise_if_not_all_None(list_parameters):
    """ if one is not None, throws an error"""
    for parameter in list_parameters:
        if not is_None(parameter):
            raise ValueError(str(parameter))  # wip is it raising the way it should?
    return


def nn_fit(net, X_train_on_device, Y_train_on_device, Y_train,
           params_training, training_losses, training_accuracy,
           X_val_on_device=None, Y_val_on_device=None, Y_val=None,
           validation_losses=None, validation_accuracy=None,
           max_through_epoch=None,
           early_stopper_training=None, early_stopper_validation=None, silent=False):
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
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze_()

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

        # Normalize and save the loss over the current epoch
        training_losses[epoch] = train_loss / (Y_train_on_device.shape[0])
        training_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_train_on_device),
                                                                  Y_train)  # sklearn can't access data on gpu.

        # Calculate validation loss for the current epoch
        if is_validation_included:
            # Y_val_on_device = Y_val_on_device.squeeze_()
            validation_losses[epoch] = criterion(net(X_val_on_device), Y_val_on_device.squeeze_()).item()
            validation_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_val_on_device),
                                                                        Y_val)  # sklearn can't access data on gpu.

            # Calculations to see if it's time to stop early
            if early_stopper_validation is not None:
                if early_stopper_validation(validation_losses, epoch):
                    break  # get out of epochs.
        if early_stopper_training is not None:
            if early_stopper_training(training_losses, epoch, net):
                break  # get out of epochs.

    #### end of the for in epoch.
    # we change the value of max_through_epoch:
    # wip
    max_through_epoch = epoch


def nn_train(net, data_X, data_Y,
             params_training,
             indic_train_X, indic_train_Y,
             indices_validation_X=None, indices_validation_Y=None,
             early_stopper_training=None, early_stopper_validation=None,
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
        indices_validation_X:
        indices_validation_Y:
        early_stopper_training: early_stopper_training type,
        early_stopper_validation: early_stopper_validation type,
        silent:

    Returns: Trained net.

    """
    max_through_epoch = 0  # : nb of epochs that the NN has back propagated over.

    # Prepare Training set
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train = data_Y[indic_train_Y]  # : useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    # prepare for iteration over epochs:
    training_losses = np.zeros(params_training.epochs)
    training_accuracy = np.zeros(params_training.epochs)

    # condition if we use validation set:
    list_params_validation = [indices_validation_X, indices_validation_Y]
    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to are all not None ?
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indices_validation_X].to(device)
        Y_val = data_Y[indices_validation_Y]  # :useful for using it in order to compute accuracy.
        Y_val_on_device = Y_val.to(device)
        validation_losses = np.zeros(params_training.epochs)
        validation_accuracy = np.zeros(params_training.epochs)

        # wip MAX_THROUGH_EPOCH is it working correctly?
        # essentially, we need to check what is the max epoch.
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses,
               training_accuracy, X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, Y_val=Y_val,
               validation_losses=validation_losses, validation_accuracy=validation_accuracy,
               max_through_epoch=max_through_epoch,
               early_stopper_training=early_stopper_training, early_stopper_validation=early_stopper_validation,
               silent=silent)

        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch)
        else:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch)

    else:
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses,
               training_accuracy, max_through_epoch=max_through_epoch, early_stopper_training=early_stopper_training,
               early_stopper_validation=early_stopper_validation, silent=silent)
        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy,
                    training_losses,
                    max_through_epoch)
        else:
            return (training_accuracy,
                    training_losses,
                    max_through_epoch)


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


# create main cross validation method
# it returns the score during training, but also the best out of the k models, with respect to the accuracy over the whole set.
def nn_kfold_train(data_training_X, data_training_Y, input_size, hidden_sizes, output_size, biases,
                   activation_functions, p, parameters_for_training=None, early_stopper_validation=None,
                   early_stopper_training=None, nb_split=5, shuffle=True, silent=False):
    """

    Args:
        early_stopper_training:
        data_training_X: tensor
        data_training_Y: tensor
        input_size:
        hidden_sizes:
        output_size:
        biases:
        activation_functions:
        p:
        parameters_for_training:
        early_stopper_validation:weight
        nb_split:
        shuffle:
        silent:

    Returns:

    """
    mean_training_accuracy = np.zeros((nb_split, parameters_for_training.epochs))
    mean_validation_accuracy = np.zeros((nb_split, parameters_for_training.epochs))
    mean_training_losses = np.zeros((nb_split, parameters_for_training.epochs))
    mean_validation_losses = np.zeros((nb_split, parameters_for_training.epochs))

    # The case nb_split = 1: we use the whole dataset for training, without validation:
    if nb_split == 1:
        neural_network = Fully_connected_NN(input_size, hidden_sizes,
                                            output_size, biases,
                                            activation_functions, p).to(device)
        # initialise weights of the NN:
        neural_network.init_weights_of_model()

        res = nn_train(neural_network, data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_for_training, indic_train_X=range(data_training_X.shape[0]),
                       indic_train_Y=range(data_training_Y.shape[0]), early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation, silent=silent)

        mean_training_accuracy += res[0]
        mean_training_losses += res[1]
        nb_of_epochs_through = res[2]  # :this number is how many epochs the NN has been trained over.
        # :with such quantity, one does not return a vector full of 0, but only the meaningful data.

        return (neural_network,
                mean_training_accuracy[:nb_of_epochs_through],
                mean_training_losses[:nb_of_epochs_through])

    # Kfold:
    skfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle, random_state=0)
    # for storing the newtork:
    performance = 0
    best_net = 0
    nb_max_epochs_through = 0  # :this number is how many epochs the NN has been trained over.
    # :with such quantity, one does not return a vector full of 0, but only the meaningful data.

    for i, (tr, te) in enumerate(
            skfold.split(data_training_X, data_training_Y)):  # one can use tensors as they are convertible to numpy.
        if not silent:
            print(f"{i + 1}-th Fold out of {nb_split} Folds.")

        neural_network = Fully_connected_NN(input_size, hidden_sizes, output_size, biases, activation_functions, p).to(
            device)
        # train network and save results
        res = nn_train(neural_network, data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_for_training, indic_train_X=tr, indic_train_Y=tr,
                       indices_validation_X=te, indices_validation_Y=te, early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation, silent=silent)

        mean_training_accuracy[i, :] += res[0]
        mean_validation_accuracy[i, :] += res[1]
        mean_training_losses[i, :] += res[2]
        mean_validation_losses[i, :] += res[3]
        nb_of_epochs_through = res[4]

        # storing the best network.
        new_res = res[1][-1]  # the criteria is best validation accuracy at final time.
        if new_res > performance:
            best_net = neural_network
        performance = new_res

        # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if nb_of_epochs_through > nb_max_epochs_through:
            nb_max_epochs_through = nb_of_epochs_through

    return (best_net,
            mean_training_accuracy[:, :nb_max_epochs_through], mean_validation_accuracy[:, :nb_max_epochs_through],
            mean_training_losses[:, :nb_max_epochs_through], mean_validation_losses[:, :nb_max_epochs_through])


def pytorch_device_setting(type="cpu"):
    """
    Semantics : sets the device for NeuralNetwork computations.
    Put nothing for automatic choice.
    Args:
        type:

    Returns:

    """
    device = torch_device("cpu") if type == "cpu" else device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    return device
