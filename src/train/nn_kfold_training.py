import time
import numpy as np
import sklearn.model_selection
import torch

from nn_classes.estimator.estim_history import Estim_history
from src.nn_classes.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla
from src.train.nn_train import nn_train


def nn_kfold_train(data_training_X, data_training_Y,
                   Model_NN, parameters_training,
                   early_stoppers=(Early_stopper_vanilla(),),
                   nb_split=5, shuffle_kfold=True,
                   percent_validation_for_1_fold=20, only_best_history=False, silent=False,
                   train_param_dict={}.copy()):
    """
    # create main cross validation method
    # it returns the score during training,
    # but also the best out of the nb_split models, with respect to the loss over the whole set.


    Args:
        train_param_dict: a dictionary with all the training parameters
        data_training_X: tensor
        data_training_Y: tensor
        Model_NN: parametrised architecture,
            type the Class with architecture we want to KFold over.
            Requirements: call constructor over it to create a net.
        parameters_training: NNTrainParameters. contains the parameters used for training
        early_stoppers: iterable of Early_stopper. Used for deciding if the training should stop early.
            Preferably immutable to insure no changes.
        nb_split:
        shuffle_kfold:
        percent_validation_for_1_fold:
        only_best_history:
        silent:

    Returns: net, history_kfold, best_epoch_for_model.

        history_kfold has the form:
            history = {'training': {},'validation': {}}
            history['training']['loss'] = np.zeros((nb_split, parameters_training.epochs))
            history['validation']['loss'] = np.zeros((nb_split, parameters_training.epochs))
            for metric in parameters_training.metrics:
                history['training'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
                history['validation'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
    best_epoch_for_model looks like: [10,200,5]

    Post-condition :
        early_stoppers not changed.
    """
    # place where logs of trainings with respect to the metrics are stored.
    indices, compute_validation = _nn_kfold_indices_creation_random(data_training_X,
                                                                    data_training_Y,
                                                                    percent_validation_for_1_fold,
                                                                    nb_split,
                                                                    shuffle_kfold)
    # initialise estimator
    estimator_history = _initialise_estimator(compute_validation, parameters_training, train_param_dict)

    return _nn_multiplefold_train(data_training_X, data_training_Y, early_stoppers, Model_NN, nb_split,
                                  parameters_training, indices, silent, estimator_history,
                                  only_best_history)


def _initialise_estimator(compute_validation, parameters_training, train_param_dict):
    metric_names = [metric.name for metric in parameters_training.metrics]
    estimator_history = Estim_history(metric_names=metric_names, validation=compute_validation,
                                      training_parameters=train_param_dict)
    return estimator_history


def create_history_kfold(compute_validation, early_stoppers, nb_split, parameters_training):
    """ nb of split at least 1, 1 means no split.  """
    # todo change parameters_training for params_training
    history_kfold = {'training': {}}
    history_kfold['training']['loss'] = np.zeros((nb_split, parameters_training.epochs))
    for metric in parameters_training.metrics:
        history_kfold['training'][metric.name] = np.zeros((nb_split, parameters_training.epochs))
    if compute_validation:
        history_kfold['validation'] = {}
        history_kfold['validation']['loss'] = np.zeros((nb_split, parameters_training.epochs))
        for metric in parameters_training.metrics:
            history_kfold['validation'][metric.name] = np.zeros((nb_split, parameters_training.epochs))

    else:  # : testing that no early validation stopper given.
        for stop in early_stoppers:
            assert not stop.is_validation(), "Input validation stopper while no validation set given."
    return history_kfold


# section ######################################################################
#  #############################################################################
# MULTIFOLD

def _nn_multiplefold_train(data_training_X, data_training_Y, early_stoppers, Model_NN, nb_split,
                           parameters_training, indices, silent, estimator_history,
                           only_best_history=False):
    # for storing the network:
    value_metric_for_best_NN = - np.Inf  # :we set -\infty which can only be improved.
    # :Recall, the two criterea are either accuracy (so any accuracy is better than a neg. number)
    # : and minus loss, and a loss is always closer to zero than - infinity.
    best_net = None
    number_kfold_best_net = 0  # to keep track of best net

    # : random_state is the seed of StratifiedKFold.
    for i, (index_training, index_validation) in enumerate(indices):
        if not silent:
            time.sleep(0.0001)  # for printing order
            print(f"{i + 1}-th Fold out of {nb_split} Folds.")
            time.sleep(0.0001)  # for printing order

        # : one can use tensors as they are convertible to numpy.
        (best_net, value_metric_for_best_NN,
         number_kfold_best_net) = train_kfold_a_fold_after_split(data_training_X, data_training_Y,
                                                                 index_training, index_validation,
                                                                 Model_NN, parameters_training,
                                                                 estimator_history,
                                                                 early_stoppers, value_metric_for_best_NN,
                                                                 number_kfold_best_net, best_net, i, silent)

    if not silent:
        print("Finished the K-Fold Training, the best NN is the number {}".format(number_kfold_best_net + 1))

    if only_best_history:
        estimator_history.take_best_fold(number_kfold_best_net)


    return best_net, estimator_history


def train_kfold_a_fold_after_split(data_training_X, data_training_Y, index_training, index_validation, Model_NN,
                                   parameters_training, estimator_history,
                                   early_stoppers=(Early_stopper_vanilla(),), value_metric_for_best_NN=-np.Inf,
                                   number_kfold_best_net=1, best_net=None, i=0, silent=False):
    """

    Args:
        data_training_X:
        data_training_Y:
        index_training: format such that it is possible to slice data like: data[index]
        index_validation:
        Model_NN:
        parameters_training:
        early_stoppers: a list of early_stoppers
        value_metric_for_best_NN:
        number_kfold_best_net:
        best_net: can be pass None, is it for comparison. Otherwise, should be a net.
        i (unsigned int): number for positions in list. Should correspond to the iterable in range(nb_of_split)
        silent:

    Returns:
        best_net, number_kfold_best_net

    Post-conditions:
        history_kfold is updated to contain the training.
        early stoppers are not modified.
        value_metric_for_best_NN is modified.
        number_kfold_best_net is updated.
        best_net is modified for the new net.
        best_epoch_of_NN is modified.
        i is not modified.

    """
    net = Model_NN().to(parameters_training.device)

    # reset the early stoppers for the following fold
    for early_stopper in early_stoppers:
        early_stopper.reset()

    # train network and save results
    kfold_history, kfold_best_epoch = nn_train(net, data_X=data_training_X, data_Y=data_training_Y,
                                               params_training=parameters_training,
                                               indic_train_X=index_training, indic_train_Y=index_training,
                                               early_stoppers=early_stoppers,
                                               indic_validation_X=index_validation, indic_validation_Y=index_validation,
                                               silent=silent)

    estimator_history.append_history(kfold_history, kfold_best_epoch, i)

    return _new_best_model(best_net, i, net, value_metric_for_best_NN, estimator_history, number_kfold_best_net, silent)


def _new_best_model(best_net, i, net, value_metric_for_best_NN, estimator_history,
                    number_kfold_best_net, silent):
    rookie_perf = -estimator_history.get_value_at_index(i, estimator_history.best_epoch[i], "loss_training")

    if not silent:  # -1 * ... bc we want to keep order below :
        print("New best model updated: rookie perf : {}"
              " and old best perf : {}.".format(-rookie_perf, -value_metric_for_best_NN))
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

    # save the best epoch for the fold
    best_epoch_of_NN[index] = kfold_best_epoch

    # update the history with the results from the fold training
    for metric_key in kfold_history['training']:
        history['training'][metric_key][index, :] = kfold_history['training'][metric_key]

    # if validation is performed, update the history with the results from the fold validation
    if 'validation' in kfold_history:
        for metric_key in kfold_history['validation']:
            history['validation'][metric_key][index, :] = kfold_history['validation'][metric_key]


# section ######################################################################
#  #############################################################################
# INDICES

def _nn_kfold_indices_creation_random(data_training_X, data_training_Y,
                                      percent_validation_for_1_fold,
                                      nb_split, shuffle_kfold):
    """ returns a list of tuples (a tuple per fold) and a bool = compute_validation"""
    # Only one fold
    if nb_split == 1:
        assert 0 <= percent_validation_for_1_fold < 100, "percent_validation_for_1_fold should be in [0,100[ !"

        # Without validation fold
        if percent_validation_for_1_fold == 0:
            return [(torch.arange(data_training_X.shape[0]), None)], False
            # : kfold split hands back list of tuples. List container, a tuple for each fold.

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
        # : kfold split hands back list of tuples. List container, a tuple for each fold.

    # multiple folds
    else:
        try:
            # classification
            kfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold)  # set seed higher

            # attempt to use the indices to check whether we can use stratified kfold
            for _ in kfold.split(data_training_X, data_training_Y):
                break

        except ValueError:
            # regression
            kfold = sklearn.model_selection.KFold(n_splits=nb_split, shuffle=shuffle_kfold)  # set seed higher

        return kfold.split(data_training_X, data_training_Y), True
