import time

import numpy as np
import sklearn.model_selection
import torch

from src.train.NN_train import nn_train
from src.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla


# todo untangle kfold with training.


def nn_kfold_train(data_training_X, data_training_Y,
                   model_NN, parameters_training,
                   early_stoppers=(Early_stopper_vanilla()),
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

    history = {
        "training": {},
        "validation": {}
    }
    history["training"]["loss"] = np.zeros((nb_split, parameters_training.epochs))
    history["validation"]["loss"] = np.zeros((nb_split, parameters_training.epochs))

    for metric in parameters_training.metrics:
        history["training"][metric] = np.zeros((nb_split, parameters_training.epochs))
        history["validation"][metric] = np.zeros((nb_split, parameters_training.epochs))


    indices, compute_validation = _nn_kfold_indices_creation(data_training_X,
                                                             data_training_Y,
                                                             percent_validation_for_1_fold,
                                                             nb_split,
                                                             shuffle_kfold)

    return _nn_multiplefold_train(compute_accuracy,
                                  data_training_X, data_training_Y,
                                  early_stoppers,
                                  model_NN, nb_split, parameters_training, indices,
                                  silent,
                                  history)


# section ######################################################################
#  #############################################################################
# MULTIFOLD

def _nn_multiplefold_train(compute_accuracy,
                           data_training_X, data_training_Y,
                           early_stoppers,
                           model_NN, nb_split, parameters_training, indices,
                           silent,
                           history):

    # for storing the network:
    value_metric_for_best_NN = - np.Inf  # :we set -\infty which can only be improved.
    # :Recall, the two criterea are either accuracy (so any accuracy is better than a neg. number)
    # : and minus loss, and a loss is always closer to zero than - infinity.
    best_net = None
    number_kfold_best_net = 1  # to keep track of best net
    best_epoch_of_NN = [0] * nb_split  # :we store the epoch of the best net for each fold.

    # : random_state is the seed of StratifiedKFold.
    for i, (index_training, index_validation) in enumerate(indices):
        # : one can use tensors as they are convertible to numpy.
        best_net, number_kfold_best_net = train_kfold_a_fold_after_split(best_epoch_of_NN, best_net,
                                                                         compute_accuracy,
                                                                         data_training_X, data_training_Y,
                                                                         early_stoppers, i,
                                                                         index_training, index_validation, model_NN,
                                                                         nb_split, number_kfold_best_net,
                                                                         parameters_training,
                                                                         value_metric_for_best_NN, silent,
                                                                         history)

    if not silent:
        print("Finis the K-Fold, the best NN is the number {}".format(number_kfold_best_net))

    return best_net, history, best_epoch_of_NN


def train_kfold_a_fold_after_split(best_epoch_of_NN, best_net, compute_accuracy,
                                   data_training_X, data_training_Y,
                                   early_stoppers, i, index_training,
                                   index_validation, model_NN, nb_split, number_kfold_best_net, parameters_training,
                                   value_metric_for_best_NN, silent, history):
    if not silent:
        time.sleep(0.001)  # for printing order
        print(f"{i + 1}-th Fold out of {nb_split} Folds.")
        time.sleep(0.001)  # for printing order

    net = model_NN().to(parameters_training.device)
    for early_stopper in early_stoppers:
        early_stopper.reset()

    # train network and save results
    res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y, params_training=parameters_training,
                   indic_train_X=index_training, indic_train_Y=index_training,
                   early_stoppers=early_stoppers,
                   indic_validation_X=index_validation, indic_validation_Y=index_validation,
                   compute_accuracy=compute_accuracy, silent=silent)

    _set_history_from_nn_train(res=res, best_epoch_of_NN=best_epoch_of_NN, history=history, index=i)

    # storing the best network.
    best_net, value_metric_for_best_NN, number_kfold_best_net = _new_best_model(best_epoch_of_NN, best_net,
                                                                                i, net,
                                                                                value_metric_for_best_NN,
                                                                                history, number_kfold_best_net)
    return best_net, number_kfold_best_net


def _new_best_model(best_epoch_of_NN, best_net, i, net, value_metric_for_best_NN, history, number_kfold_best_net):

    rookie_perf = - history["validation"]["loss"][i, best_epoch_of_NN[i]]  #: -1 * ... bc we want to keep order below
    if value_metric_for_best_NN < rookie_perf:
        best_net = net
        value_metric_for_best_NN = rookie_perf
        number_kfold_best_net = i
    return best_net, value_metric_for_best_NN, number_kfold_best_net


# section ######################################################################
#  #############################################################################
# HISTORY FUNCTION
def _set_history_from_nn_train(res, best_epoch_of_NN, history, index):
    kfold_history, kfold_best_epoch = res
    best_epoch_of_NN[index] = kfold_best_epoch

    for metric_key in kfold_history["training"]:
        history["training"][metric_key][index, :] = kfold_history["training"][metric_key]

    if "validation" in kfold_history:
        for metric_key in kfold_history["validation"]:
            history["validation"][metric_key][index, :] = kfold_history["validation"][metric_key]


# section ######################################################################
#  #############################################################################
# INDICES

def _nn_kfold_indices_creation(data_training_X, data_training_Y, percent_validation_for_1_fold, nb_split,
                               shuffle_kfold):
    # Only one fold
    if nb_split == 1:
        assert 0 <= percent_validation_for_1_fold < 100, "percent_validation_for_1_fold should be in [0,100[ !"

        # Without validation fold
        if percent_validation_for_1_fold == 0:
            return [(torch.arange(data_training_X.shape[0]), None)], False

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
            indic_validation = torch.arange(training_size, data_training_X.shape[0])
        return [(indic_train, indic_validation)], True

    else:
        try:
            kfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)
        except ValueError:
            kfold = sklearn.model_selection.KFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)

        return kfold.split(data_training_X, data_training_Y), True
