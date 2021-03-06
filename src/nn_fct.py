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

from src.NeuralNet import *
from src.NNTrainParameters import *
from src.plots import *
from src.kfold import *
from src.EarlyStopping import *


def nn_train(neural_network,
             data_X, data_Y,
             indices_train_X, indices_train_Y,
             indices_validation_X=None, indices_validation_Y=None,
             parameters_for_training=None, early_stopper=None,
             silent=False):
    X = data_X.iloc[indices_train_X].values
    y = data_Y.iloc[indices_train_Y].values
    validation_X = data_X.iloc[indices_validation_X].values
    validation_Y = data_Y.iloc[indices_validation_Y].values

    is_validation_included = (validation_X is not None and validation_Y is not None)

    # Prepare Validation set if there is any
    if is_validation_included:
        X_val = torch.from_numpy(validation_X).float().to(device)
        Y_val = torch.from_numpy(validation_Y).long().to(device)
        validation_losses = np.zeros(parameters_for_training.epochs)
        validation_accuracy = np.zeros(parameters_for_training.epochs)

    # prepare for iteration over epochs
    training_losses = np.zeros(parameters_for_training.epochs)
    training_accuracy = np.zeros(parameters_for_training.epochs)

    # Prepare Training set and create data loader
    X_train = torch.from_numpy(X).float()
    Y_train = torch.from_numpy(y).long()
    nn_train = torch.utils.data.TensorDataset(X_train, Y_train)

    loader = torch.utils.data.DataLoader(nn_train, batch_size=parameters_for_training.batch_size, shuffle=True)
    # pick loss function and optimizer
    criterion = parameters_for_training.criterion
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    optimiser = parameters_for_training.optimiser
    # sgd = torch.optim.SGD(Neural_Network.parameters(), lr=parameters_for_training.learning_rate)

    for epoch in range(parameters_for_training.epochs):
        train_loss = 0
        for i, batch in enumerate(loader, 0):
            # get batch
            batch_X, batch_y = batch
            # set gradients to zero
            optimiser.zero_grad()
            # Do forward and backward pass
            out = neural_network(batch_X)
            batch_y = batch_y.squeeze_()  # not the good size for the results
            loss = criterion(out, batch_y)
            loss.backward()
            # Optimisation step
            optimiser.step()

            train_loss += loss.item()
        if epoch % 10 == 1 and not silent:
            print(epoch, "Epochs complete")

        # Normalize and save the loss over the current epoch
        training_losses[epoch] = train_loss * parameters_for_training.batch_size / (y.shape[0])
        training_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(neural_network, X), y)

        # Calculate validation loss for the current epoch
        if is_validation_included:
            # Y_val = Y_val.squeeze_()
            validation_losses[epoch] = criterion(neural_network(X_val), Y_val).item()
            validation_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(neural_network, validation_X),
                                                                        validation_Y)
            # Calculations to see if it's time to stop early
            if early_stopper is not None:
                if early_stopper(training_losses, validation_losses, epoch, neural_network):
                    break  # get out of epochs.

    #### end of the for in epoch.
    if not silent:
        print("Training done after {} epochs ! ".format(parameters_for_training.epochs))

    # return loss over epochs and accuracy
    if is_validation_included:
        if early_stopper is not None:
            return training_accuracy, validation_accuracy, training_losses, validation_losses, epoch
        else:
            return training_accuracy, validation_accuracy, training_losses, validation_losses
    else:
        if early_stopper is not None:
            return training_accuracy, training_losses, epoch
        else:
            return training_accuracy, validation_accuracy, training_losses, validation_losses


def nn_predict(NeuralNetwork, data_to_predict):
    # do a single predictive forward pass on pnet (takes & returns numpy arrays)
    # Disable dropout:
    NeuralNetwork.train(mode=False)
    # forward pass
    data_predicted = NeuralNetwork(data_to_predict).float().detach().numpy()

    # WIP
    # yhat = pnet(torch.from_numpy(pX.values).float().to(device)).argmax(dim=1).cpu().numpy()
    # WIP

    # Reable dropout:
    NeuralNetwork.train(mode=True)
    return data_predicted


# create main cross validation method
# it returns the score during training, but also the best out of the k models, with respect to the accuracy over the whole set.
def nn_kfold_train(
        data_training_X, data_training_Y,
        input_size, hidden_sizes, output_size, biases, activation_functions, p,
        parameters_for_training=None, early_stopper=None,
        nb_split=5, shuffle=True,
        silent=False):
    mean_training_accuracy = np.zeros(parameters_for_training.epochs)
    mean_valid_accuracy = np.zeros(parameters_for_training.epochs)
    mean_train_losses = np.zeros(parameters_for_training.epochs)
    mean_valid_losses = np.zeros(parameters_for_training.epochs)

    # The case nb_split = 1: we use the whole dataset for training, without validation.
    if nb_split == 1:
        neural_network = NeuralNet(input_size, hidden_sizes, output_size, biases, activation_functions, p).to(device)
        res = nn_train(neural_network,
                       data_X=data_training_X, data_Y=data_training_Y,
                       indices_train_X=[], indices_train_Y=[],
                       parameters_for_training=parameters_for_training, early_stopper=early_stopper,
                       silent=silent)

        mean_training_accuracy += res[0]
        mean_train_losses += res[1]
        nb_of_epochs_through = res[2] # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.

        return (neural_network,
                mean_training_accuracy[:nb_of_epochs_through], mean_valid_accuracy[:nb_of_epochs_through],
                mean_train_losses[:nb_of_epochs_through], mean_valid_losses[:nb_of_epochs_through])

    # Kfold
    cv = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle, random_state=0)
    # for storing the newtork
    performance = 0
    best_net = 0
    nb_max_epochs_through = 0  # this number is how many epochs the NN has been trained over.
    # with such quantity, one does not return a vector full of 0, but only the meaningful data.

    for i, (tr, te) in enumerate(cv.split(data_training_X, data_training_Y)):
        if not silent:
            print(f"{i}-th Fold out of {nb_split} Folds.")

        neural_network = NeuralNet(input_size, hidden_sizes, output_size, biases, activation_functions, p).to(device)
        # train network and save results
        res = nn_train(neural_network,
                       data_X=data_training_X, data_Y=data_training_Y,
                       indices_train_X=tr, indices_train_Y=tr,
                       indices_validation_X=te, indices_validation_Y=te,
                       parameters_for_training=parameters_for_training, early_stopper=early_stopper,
                       silent=silent)

        mean_training_accuracy += res[0]
        mean_valid_accuracy += res[1]
        mean_train_losses += res[2]
        mean_valid_losses += res[3]
        nb_of_epochs_through = res[4]

        # storing the network :
        new_res = nn_predict(neural_network, data_training_X)
        new_res = sklearn.metrics.accuracy_score(new_res, data_training_Y)
        if new_res > performance:
            best_net = neural_network
        performance = new_res

        # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if nb_of_epochs_through > nb_max_epochs_through:
            nb_max_epochs_through = nb_of_epochs_through

    # Average and return results
    mean_training_accuracy /= nb_split
    mean_valid_accuracy /= nb_split
    mean_train_losses /= nb_split
    mean_valid_losses /= nb_split
    return (best_net,
           mean_training_accuracy[:nb_max_epochs_through], mean_valid_accuracy[:nb_max_epochs_through],
           mean_train_losses[:nb_max_epochs_through], mean_valid_losses[:nb_max_epochs_through])
