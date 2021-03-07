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
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") #forces the usage of cpu

def nn_train(net, data_X, data_Y, indices_train_X, indices_train_Y, indices_validation_X=None,
             indices_validation_Y=None, parameters_for_training=None, early_stopper=None, silent=False):
    """
    Semantics:
        Given the net, we train it upon data. For optimisation reasons, we pass the indices.

    Args:
        net:
        data_X: tensor
        data_Y: tensor
        indices_train_X:
        indices_train_Y:
        indices_validation_X:
        indices_validation_Y:
        parameters_for_training:
        early_stopper:
        silent:

    Returns: trained net.

    """
    # Prepare Training set and create data loader
    X_train_on_device = data_X[indices_train_X].to(device)
    Y_train = data_Y[indices_train_Y] # useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    data_training = torch.utils.data.TensorDataset(X_train_on_device, Y_train_on_device)
    loader = torch.utils.data.DataLoader(data_training,
                                         batch_size=parameters_for_training.batch_size,
                                         shuffle=True,
                                         num_workers= 0) #num_workers can be increased

    # condition if we use validation set.
    is_validation_included = (indices_validation_X is not None and indices_validation_Y is not None)
    # Prepare Validation set if there is any
    if is_validation_included:
        X_val_on_device = data_X[indices_validation_X].to(device)
        Y_val = data_Y[indices_validation_Y]  # useful for using it in order to compute accuracy.
        Y_val_on_device = Y_val.to(device)
        validation_losses = np.zeros(parameters_for_training.epochs)
        validation_accuracy = np.zeros(parameters_for_training.epochs)

    # prepare for iteration over epochs
    training_losses = np.zeros(parameters_for_training.epochs)
    training_accuracy = np.zeros(parameters_for_training.epochs)


    # pick loss function and optimizer
    criterion = parameters_for_training.criterion
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    optimiser = parameters_for_training.optimiser(net.parameters(), lr= parameters_for_training.learning_rate )


    for epoch in tqdm(range(parameters_for_training.epochs), disable = silent):
        train_loss = 0
        for i, (batch_X, batch_y) in enumerate(loader, 0):
            # get batch
            # squeeze batch y in order to have the right format. not the good size for the results
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze_()

            # set gradients to zero
            optimiser.zero_grad()

            # Do forward and backward pass
            loss = criterion(net(batch_X), batch_y)
            loss.backward()

            # Optimisation step
            optimiser.step()

            train_loss += loss.item() * batch_X.shape[0] # weight the loss accordingly

        # if epoch % 10 == 1 and not silent:
        #     print(epoch, "Epochs complete")

        # Normalize and save the loss over the current epoch
        training_losses[epoch] = train_loss  / (Y_train_on_device.shape[0])
        training_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_train_on_device), Y_train)  # sklearn can't access data on gpu.

        # Calculate validation loss for the current epoch
        if is_validation_included:
            # Y_val_on_device = Y_val_on_device.squeeze_()
            validation_losses[epoch] = criterion(net(X_val_on_device), Y_val_on_device.squeeze_()).item()
            validation_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_val_on_device), Y_val) # sklearn can't access data on gpu.
            # Calculations to see if it's time to stop early
            if early_stopper is not None:
                if early_stopper(training_losses, validation_losses, epoch, net):
                    break  # get out of epochs.
    #### end of the for in epoch.


    # return loss over epochs and accuracy
    if is_validation_included:
        if early_stopper is not None:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    epoch)
        else:
            return (training_accuracy, validation_accuracy,
                   training_losses, validation_losses,
                    parameters_for_training.epochs - 1) # the -1 is because epochs start at 0.
    else:
        if early_stopper is not None:
            return training_accuracy, training_losses, epoch
        else:
            return (training_accuracy,
                    training_losses,
                    parameters_for_training.epochs -1)  # the -1 is because epochs start at 0.


def nn_predict(net, data_to_predict):
    # do a single predictive forward pass on net (takes & returns numpy arrays)
    # Disable dropout:
    net.train(mode=False)

    # forward pass
    # to device for optimal speed, though we take the data back with .cpu().
    data_predicted = net.prediction(net(data_to_predict.to(device))).detach().cpu()

    # Re-able dropout:
    net.train(mode=True)
    return data_predicted


# create main cross validation method
# it returns the score during training, but also the best out of the k models, with respect to the accuracy over the whole set.
def nn_kfold_train(
        data_training_X, data_training_Y,
        input_size, hidden_sizes, output_size, biases, activation_functions, p,
        parameters_for_training=None, early_stopper=None,
        nb_split=5, shuffle=True,
        silent=False):
    """

    Args:
        data_training_X: tensor
        data_training_Y: tensor
        input_size:
        hidden_sizes:
        output_size:
        biases:
        activation_functions:
        p:
        parameters_for_training:
        early_stopper:
        nb_split:
        shuffle:
        silent:

    Returns:

    """
    mean_training_accuracy = np.zeros((nb_split, parameters_for_training.epochs))
    mean_validation_accuracy = np.zeros((nb_split, parameters_for_training.epochs))
    mean_training_losses = np.zeros((nb_split, parameters_for_training.epochs))
    mean_validation_losses = np.zeros((nb_split, parameters_for_training.epochs))

    # The case nb_split = 1: we use the whole dataset for training, without validation.
    if nb_split == 1:
        neural_network = NeuralNet(input_size, hidden_sizes, output_size, biases, activation_functions, p).to(device)
        res = nn_train(neural_network,
                       data_X=data_training_X, data_Y=data_training_Y,
                       indices_train_X=range(data_training_X.shape[0]),
                       indices_train_Y=range(data_training_Y.shape[0]), parameters_for_training=parameters_for_training, early_stopper=early_stopper,
                       silent=silent)


        mean_training_accuracy += res[0]
        mean_training_losses += res[1]
        nb_of_epochs_through = res[2]  # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.

        return (neural_network,
                mean_training_accuracy[:nb_of_epochs_through],
                mean_training_losses[:nb_of_epochs_through])

    # Kfold
    skfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle, random_state=0)
    # for storing the newtork
    performance = 0
    best_net = 0
    nb_max_epochs_through = 0  # this number is how many epochs the NN has been trained over.
    # with such quantity, one does not return a vector full of 0, but only the meaningful data.

    for i, (tr, te) in enumerate(skfold.split(data_training_X, data_training_Y)): # one can use tensors as they are convertible to numpy.
        if not silent:
            print(f"{i+1}-th Fold out of {nb_split} Folds.")

        neural_network = NeuralNet(input_size, hidden_sizes, output_size, biases, activation_functions, p).to(device)
        # train network and save results
        res = nn_train(neural_network, data_X=data_training_X, data_Y=data_training_Y, indices_train_X=tr,
                       indices_train_Y=tr, indices_validation_X=te, indices_validation_Y=te,
                       parameters_for_training=parameters_for_training, early_stopper=early_stopper, silent=silent)

        mean_training_accuracy[i,:] += res[0]
        mean_validation_accuracy[i, :] += res[1]
        mean_training_losses[i, :] += res[2]
        mean_validation_losses[i, :] += res[3]
        nb_of_epochs_through = res[4]

        #storing the best network.
        new_res = res[1][-1] # the criteria is best validation accuracy at final time.
        if new_res > performance:
            best_net = neural_network
        performance = new_res

        # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if nb_of_epochs_through > nb_max_epochs_through:
            nb_max_epochs_through = nb_of_epochs_through

    return (best_net,
            mean_training_accuracy[:, :nb_max_epochs_through], mean_validation_accuracy[:,:nb_max_epochs_through],
            mean_training_losses[:,:nb_max_epochs_through], mean_validation_losses[:,:nb_max_epochs_through])
