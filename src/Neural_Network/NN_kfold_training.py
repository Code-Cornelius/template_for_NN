import numpy as np
import sklearn.model_selection
import torch

from src.Neural_Network.NN_fcts import device
from src.Neural_Network.NN_training import nn_train


def nn_kfold_train(data_training_X, data_training_Y,
                   model_NN, parameters_training=None,
                   early_stopper_validation=None,
                   early_stopper_training=None,
                   nb_split=5, shuffle_kfold=True,
                   percent_validation_for_1_fold=20,
                   compute_accuracy=False,
                   silent=False):
    """
    # create main cross validation method
    # it returns the score during training,
    # but also the best out of the k models, with respect to the accuracy over the whole set.

    Args:
        model_NN: parametrised architecture,
        should be a Class (object type Class) and such that we can call constructor over it to create a net.
        compute_accuracy:
        early_stopper_training:
        data_training_X: tensor
        data_training_Y: tensor
        parameters_training:
        early_stopper_validation:weight
        nb_split:
        shuffle_kfold:
        percent_validation_for_1_fold:
        silent:

    Returns: net, loss train, loss validation, accuracy train, accuracy validation

    """
    # we distinguish the two cases, but in both we have a list of the result:
    # by inclusivity of else into if compute_accuracy, [0] should be loss and [1] accuracy:

    # the nans are because we want to skip the plotting at places where we did not collect data.
    if compute_accuracy:
        training_data = [np.zeros((nb_split, parameters_training.epochs)),
                         np.zeros((nb_split, parameters_training.epochs))]
        validation_data = [np.zeros((nb_split, parameters_training.epochs)),
                           np.zeros((nb_split, parameters_training.epochs))]
    else:
        training_data = [np.zeros((nb_split, parameters_training.epochs))]
        validation_data = [np.zeros((nb_split, parameters_training.epochs))]

    # The case nb_split = 1: we use the whole dataset for training, without validation:
    if nb_split == 1:
        assert 0 <= percent_validation_for_1_fold < 100, "percent_validation_for_1_fold should be in [0,100[ !"
        return _nn_1fold_train(compute_accuracy, data_training_X, data_training_Y, early_stopper_training,
                               early_stopper_validation, parameters_training, percent_validation_for_1_fold,
                               shuffle_kfold, silent, training_data, validation_data, model_NN)
    else:
        best_net, nb_max_epochs_through = _nn_multiplefold_train(compute_accuracy, data_training_X, data_training_Y,
                                                                 early_stopper_training, early_stopper_validation,
                                                                 model_NN, nb_split, parameters_training, shuffle_kfold,
                                                                 silent, training_data, validation_data)

        return (best_net,
                training_data[1][:, :nb_max_epochs_through], validation_data[1][:, :nb_max_epochs_through],
                training_data[0][:, :nb_max_epochs_through], validation_data[0][:, :nb_max_epochs_through])


def _nn_multiplefold_train(compute_accuracy, data_training_X, data_training_Y, early_stopper_training,
                           early_stopper_validation, model_NN, nb_split, parameters_training, shuffle_kfold, silent,
                           training_data, validation_data):
    # Kfold for nb_split > 1:
    skfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)
    # for storing the network:
    performance = 0
    best_net = 0
    nb_max_epochs_through = 0  # :this number is how many epochs the NN has been trained over.
    # :with such quantity, one does not return a vector full of 0, but only the meaningful data.
    # it is equal to max_{kfold} min_{epoch} { epoch trained over }
    for i, (index_training, index_validation) in enumerate(
            skfold.split(data_training_X, data_training_Y)):  # one can use tensors as they are convertible to numpy.
        if not silent:
            print(f"{i + 1}-th Fold out of {nb_split} Folds.")

        net = model_NN().to(device)

        # train network and save results
        res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_training,
                       indic_train_X=index_training, indic_train_Y=index_training,
                       indic_validation_X=index_validation,
                       indic_validation_Y=index_validation,
                       early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation,
                       compute_accuracy=compute_accuracy,
                       silent=silent)

        if compute_accuracy:
            training_data[1][i, :] = res[0]
            validation_data[1][i, :] += res[1]
        training_data[0][i, :] += res[2]
        validation_data[0][i, :] += res[3]
        nb_of_epochs_through = res[4]

        # storing the best network.
        new_res = res[1][-1]  # the criteria is best validation accuracy at final time.
        if new_res > performance:
            best_net = net
        performance = new_res

        # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if nb_of_epochs_through > nb_max_epochs_through:
            nb_max_epochs_through = nb_of_epochs_through
    return best_net, nb_max_epochs_through


def _nn_1fold_train(compute_accuracy, data_training_X, data_training_Y, early_stopper_training,
                    early_stopper_validation, parameters_training, percent_validation_for_1_fold, shuffle_kfold, silent,
                    training_data, validation_data, model_NN):
    net = model_NN().to(device)
    # where validation included.
    if percent_validation_for_1_fold > 0:
        # #indices splitting:
        indic_train, indic_validation = nn_1fold_indices_creation(data_training_X, percent_validation_for_1_fold,
                                                                  shuffle_kfold)

        res = nn_train(net,
                       data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_training,
                       indic_train_X=indic_train,
                       indic_train_Y=indic_train,
                       indic_validation_X=indic_validation,
                       indic_validation_Y=indic_validation,
                       early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation,
                       compute_accuracy=compute_accuracy,
                       silent=silent)

        training_data[0][0, :] += res[2]
        validation_data[0][0, :] += res[3]
        nb_of_epochs_through = res[4]  # :this number is how many epochs the NN has been trained over.
        # :with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if compute_accuracy:
            training_data[1] += res[0]
            validation_data[1] += res[1]
            return (net,
                    training_data[1][:, :nb_of_epochs_through], validation_data[1][:, :nb_of_epochs_through],
                    training_data[0][:, :nb_of_epochs_through], validation_data[0][:, :nb_of_epochs_through])
        return (net,
                training_data[0][:, :nb_of_epochs_through], validation_data[0][:, :nb_of_epochs_through])


    # 1-Fold. where validation not included.
    else:
        res = nn_train(net,
                       data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_training,
                       indic_train_X=torch.arange(data_training_X.shape[0]),
                       indic_train_Y=torch.arange(data_training_Y.shape[0]),
                       indic_validation_X=None,
                       indic_validation_Y=None,
                       early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation,
                       compute_accuracy=compute_accuracy,
                       silent=silent)

        training_data[0] += res[1]
        nb_of_epochs_through = res[2]  # :this number is how many epochs the NN has been trained over.
        # :with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if compute_accuracy:
            training_data[1] += res[0]
            return (net,
                    training_data[1][:nb_of_epochs_through],
                    training_data[0][:nb_of_epochs_through])
        return (net, training_data[0][:nb_of_epochs_through])
        # : do not return accuracy, only the losses.


def nn_1fold_indices_creation(data_training_X, percent_validation_for_1_fold, shuffle_kfold):
    training_size = int((100. - percent_validation_for_1_fold) / 100. * data_training_X.shape[0])
    if shuffle_kfold:
        # for the permutation, one could look at https://discuss.pytorch.org/t/shuffling-a-tensor/25422/7:
        # we simplify the expression bc our tensors are in 2D only:
        indices = torch.randperm(data_training_X.shape[0])
        #: create a random permutation of the range( nb of data )

        indic_train = indices[:training_size]
        indic_validation = indices[training_size:]
    else:
        indic_train = torch.arange(training_size)
        indic_validation = torch.arange(training_size, data_training_X.shape[1])
    return indic_train, indic_validation