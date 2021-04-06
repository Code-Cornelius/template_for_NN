import time

import numpy as np
import sklearn.model_selection
import torch

from src.Neural_Network.NN_train import nn_train
from src.Training_stopper.Early_stopper_vanilla import Early_stopper_vanilla


def nn_kfold_train(data_training_X, data_training_Y,
                   model_NN, parameters_training,
                   early_stopper_validation=Early_stopper_vanilla(),
                   early_stopper_training=Early_stopper_vanilla(),
                   nb_split=5, shuffle_kfold=True, percent_validation_for_1_fold=20,
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
        early_stopper_validation:
        nb_split:
        shuffle_kfold:
        percent_validation_for_1_fold:
        silent:

    Returns: net, loss train, loss validation, accuracy train, accuracy validation, best_epoch_for_model

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
        return _nn_multiplefold_train(compute_accuracy, data_training_X, data_training_Y,
                                      early_stopper_training, early_stopper_validation,
                                      model_NN, nb_split, parameters_training, shuffle_kfold,
                                      silent, training_data, validation_data)


# section ######################################################################
#  #############################################################################
# MULTIFOLD

def _nn_multiplefold_train(compute_accuracy, data_training_X, data_training_Y, early_stopper_training,
                           early_stopper_validation, model_NN, nb_split, parameters_training, shuffle_kfold, silent,
                           training_data, validation_data):
    # for storing the network:
    value_metric_for_best_NN = - np.Inf # :we set -\infty which can only be improved.
    # :Recall, the two criterea are either accuracy (so any accuracy is better than a neg. number)
    # : and minus loss, and a loss is always closer to zero than - infinity.
    best_net = None
    number_kfold_best_net = 1  # to keep track of best net
    best_epoch_of_NN = [0] * nb_split  # :we store the epoch of the best net for each fold.

    try:
        # Kfold for nb_split > 1:
        skfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)
        # : random_state is the seed of StratifiedKFold.
        for i, (index_training, index_validation) in enumerate(
                skfold.split(data_training_X,
                             data_training_Y)):  # one can use tensors as they are convertible to numpy.
            best_net, number_kfold_best_net = train_kfold_a_fold_after_split(best_epoch_of_NN, best_net,
                                                                             compute_accuracy, data_training_X,
                                                                             data_training_Y, early_stopper_training,
                                                                             early_stopper_validation, i,
                                                                             index_training, index_validation, model_NN,
                                                                             nb_split, number_kfold_best_net,
                                                                             parameters_training,
                                                                             value_metric_for_best_NN, silent,
                                                                             training_data, validation_data)
    except ValueError:  # erro from skfold split.
        # Kfold for nb_split > 1:
        kfold = sklearn.model_selection.KFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)
        # : random_state is the seed of StratifiedKFold.
        for i, (index_training, index_validation) in enumerate(
                kfold.split(data_training_X,
                            data_training_Y)):  # one can use tensors as they are convertible to numpy.
            best_net, number_kfold_best_net = train_kfold_a_fold_after_split(best_epoch_of_NN, best_net,
                                                                             compute_accuracy, data_training_X,
                                                                             data_training_Y, early_stopper_training,
                                                                             early_stopper_validation, i,
                                                                             index_training, index_validation, model_NN,
                                                                             nb_split, number_kfold_best_net,
                                                                             parameters_training,
                                                                             value_metric_for_best_NN, silent,
                                                                             training_data, validation_data)
    if not silent:
        print("Finis the K-Fold, the best NN is the number {}".format(number_kfold_best_net))
    if compute_accuracy:
        return (best_net, training_data[1], validation_data[1],
                training_data[0], validation_data[0], best_epoch_of_NN)
    else:
        return (best_net, training_data[0], validation_data[0], best_epoch_of_NN)


def train_kfold_a_fold_after_split(best_epoch_of_NN, best_net, compute_accuracy, data_training_X, data_training_Y,
                                   early_stopper_training, early_stopper_validation, i, index_training,
                                   index_validation, model_NN, nb_split, number_kfold_best_net, parameters_training,
                                   value_metric_for_best_NN, silent, training_data, validation_data):
    if not silent:
        time.sleep(0.001)  # for printing order
        print(f"{i + 1}-th Fold out of {nb_split} Folds.")
        time.sleep(0.001)  # for printing order

    net = model_NN().to(parameters_training.device)
    if early_stopper_training is not None:
        early_stopper_training.reset()
    if early_stopper_validation is not None:
        early_stopper_validation.reset()
    # train network and save results
    res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y, params_training=parameters_training,
                   indic_train_X=index_training, indic_train_Y=index_training,
                   early_stopper_validation=early_stopper_validation, early_stopper_training=early_stopper_training,
                   indic_validation_X=index_validation, indic_validation_Y=index_validation,
                   compute_accuracy=compute_accuracy, silent=silent)
    _set_history_from_nn_train(best_epoch_of_NN=best_epoch_of_NN,
                               compute_accuracy=compute_accuracy,
                               compute_validation=True,
                               index=i,
                               res=res,
                               training_data=training_data,
                               validation_data=validation_data)
    # storing the best network.
    best_net, value_metric_for_best_NN, number_kfold_best_net = _new_best_model(best_epoch_of_NN, best_net,
                                                                                compute_accuracy, i, net,
                                                                                value_metric_for_best_NN,
                                                                                validation_data, number_kfold_best_net)
    return best_net, number_kfold_best_net


def _new_best_model(best_epoch_of_NN, best_net, compute_accuracy, i, net, value_metric_for_best_NN, validation_data,
                    number_kfold_best_net):
    # criterion is either best validation accuracy, or lowest validation loss, at final time (best_epoch
    if compute_accuracy:
        rookie_perf = validation_data[1][i, best_epoch_of_NN[i]]
    else:
        rookie_perf = - validation_data[0][i, best_epoch_of_NN[i]]  #: -1 * ... bc we want to keep order below
    if value_metric_for_best_NN < rookie_perf:
        best_net = net
        value_metric_for_best_NN = rookie_perf
        number_kfold_best_net = i
    return best_net, value_metric_for_best_NN, number_kfold_best_net


# section ######################################################################
#  #############################################################################
# 1 FOLD

def _nn_1fold_train(compute_accuracy, data_training_X, data_training_Y, early_stopper_training,
                    early_stopper_validation, parameters_training, percent_validation_for_1_fold,
                    shuffle_kfold, silent, training_data, validation_data, model_NN):
    net = model_NN().to(parameters_training.device)
    # where validation included.
    best_epoch_of_NN = [0]
    if percent_validation_for_1_fold > 0:
        # #indices splitting:
        indic_train, indic_validation = _nn_1fold_indices_creation(data_training_X,
                                                                   percent_validation_for_1_fold,
                                                                   shuffle_kfold)

        res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y, params_training=parameters_training,
                       indic_train_X=indic_train, indic_train_Y=indic_train,
                       early_stopper_validation=early_stopper_validation, early_stopper_training=early_stopper_training,
                       indic_validation_X=indic_validation, indic_validation_Y=indic_validation,
                       compute_accuracy=compute_accuracy, silent=silent)

        _set_history_from_nn_train(best_epoch_of_NN=best_epoch_of_NN,
                                   compute_accuracy=compute_accuracy,
                                   compute_validation=True,
                                   index=0,
                                   res=res,
                                   training_data=training_data,
                                   validation_data=validation_data)

        if compute_accuracy:
            return (net, training_data[1], validation_data[1],
                    training_data[0], validation_data[0], best_epoch_of_NN)
        else:
            return net, training_data[0], validation_data[0], best_epoch_of_NN

    # 1-Fold. where validation not included.
    else:
        res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y, params_training=parameters_training,
                       indic_train_X=torch.arange(data_training_X.shape[0]),
                       indic_train_Y=torch.arange(data_training_Y.shape[0]),
                       early_stopper_validation=early_stopper_validation, early_stopper_training=early_stopper_training,
                       indic_validation_X=None, indic_validation_Y=None, compute_accuracy=compute_accuracy,
                       silent=silent)

        _set_history_from_nn_train(best_epoch_of_NN=best_epoch_of_NN, compute_accuracy=compute_accuracy,
                                   compute_validation=False, index=0, res=res,
                                   training_data=training_data, validation_data=validation_data)
        if compute_accuracy:
            return net, training_data[1], training_data[0], best_epoch_of_NN
        return net, training_data[0], best_epoch_of_NN
        # : do not return accuracy, only the losses.


# section ######################################################################
#  #############################################################################
# HISTORY FUNCTION

def _set_history_from_nn_train(best_epoch_of_NN, compute_accuracy, compute_validation, index,
                               res, training_data, validation_data):
    if compute_validation:
        if compute_accuracy:
            training_data[1][index, :] = res[0]
            validation_data[1][index, :] += res[1]
        training_data[0][index, :] += res[2]
        validation_data[0][index, :] += res[3]
        best_epoch_of_NN[index] = res[4]  # :we store the epoch of the best net for each fold.
    else:
        if compute_accuracy:
            training_data[1][index, :] = res[0]
        training_data[0][index, :] += res[1]
        best_epoch_of_NN[index] = res[2]  # :we store the epoch of the best net for each fold.


# section ######################################################################
#  #############################################################################
# INDICES

def _nn_1fold_indices_creation(data_training_X, percent_validation_for_1_fold, shuffle_kfold):
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
