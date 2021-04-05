import torch.nn.functional as F
from src.Training_stopper.Early_stopper_training import Early_stopper_training
from src.Training_stopper.Early_stopper_validation import Early_stopper_validation
from src.Neural_Network.NN_fcts import pytorch_device_setting
from src.Neural_Network.NN_kfold_training import nn_kfold_train
from src.Neural_Network.NN_plots import *
from src.Neural_Network.NN_training import *
from src.Neural_Network.NNTrainParameters import *
from src.Neural_Network.Fully_connected_NN import *


def test_no_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train, early_stop_valid,
                     nb_split, percent_validation_for_1_fold, compute_accuracy, silent, plot_xx, plot_yy, plot_yy_noisy,
                     testing_X, testing_Y):
    (net, mean_training_losses, mean_validation_losses, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y,
                                                                                           parametrized_NN,
                                                                                           parameters_training=parameters_training,
                                                                                           early_stopper_training=early_stop_train,
                                                                                           early_stopper_validation=early_stop_valid,
                                                                                           nb_split=nb_split,
                                                                                           shuffle_kfold=True,
                                                                                           percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                                                           compute_accuracy=compute_accuracy,
                                                                                           silent=silent)
    nn_plot_train_loss_acc(training_loss=mean_training_losses, validation_loss=mean_validation_losses,
                           best_epoch_of_NN=best_epoch_of_NN)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)


def test_no_accuracy_no_validation(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train,
                                   early_stop_valid, nb_split, percent_validation_for_1_fold, compute_accuracy, silent,
                                   plot_xx, plot_yy,
                                   plot_yy_noisy, testing_X, testing_Y):
    (net, mean_training_losses, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y,
                                                                   parametrized_NN,
                                                                   parameters_training=parameters_training,
                                                                   early_stopper_training=early_stop_train,
                                                                   early_stopper_validation=early_stop_valid,
                                                                   nb_split=nb_split,
                                                                   shuffle_kfold=True,
                                                                   percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                                   compute_accuracy=compute_accuracy,
                                                                   silent=silent)
    nn_plot_train_loss_acc(training_loss=mean_training_losses, validation_loss=None, best_epoch_of_NN=best_epoch_of_NN)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)


def test_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train, early_stop_valid,
                  nb_split, percent_validation_for_1_fold, compute_accuracy, silent, testing_X, testing_Y):
    (net, mean_training_accuracy, mean_validation_accuracy,
     mean_training_losses, mean_validation_losses, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y,
                                                                                      parametrized_NN,
                                                                                      parameters_training=parameters_training,
                                                                                      early_stopper_training=early_stop_train,
                                                                                      early_stopper_validation=early_stop_valid,
                                                                                      nb_split=nb_split,
                                                                                      shuffle_kfold=True,
                                                                                      percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                                                      compute_accuracy=compute_accuracy,
                                                                                      silent=silent)
    nn_plot_train_loss_acc(mean_training_losses, mean_validation_losses, mean_training_accuracy,
                           mean_validation_accuracy, best_epoch_of_NN=best_epoch_of_NN)
    # confusion_matrix_creator(train_Y, nn_predict(net, train_X), range(10), title="Training Set")
    # confusion_matrix_creator(testing_Y, nn_predict(net, testing_X), range(10), title="Test Set")


def test_accuracy_no_validation(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train,
                                early_stop_valid, nb_split, percent_validation_for_1_fold, compute_accuracy, silent,
                                testing_X, testing_Y):
    (net, mean_training_accuracy, mean_training_losses, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y,
                                                                                           parametrized_NN,
                                                                                           parameters_training=parameters_training,
                                                                                           early_stopper_training=early_stop_train,
                                                                                           early_stopper_validation=early_stop_valid,
                                                                                           nb_split=nb_split,
                                                                                           shuffle_kfold=True,
                                                                                           percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                                                           compute_accuracy=compute_accuracy,
                                                                                           silent=silent)
    nn_plot_train_loss_acc(mean_training_losses, None, mean_training_accuracy, None, best_epoch_of_NN=best_epoch_of_NN)
    # confusion_matrix_creator(train_Y, nn_predict(net, train_X), range(10), title="Training Set")
    # confusion_matrix_creator(testing_Y, nn_predict(net, testing_X), range(10), title="Test Set")


def test(train_X, train_Y, parametrized_NN, parameters_training, testing_X, testing_Y, early_stop_train,
         early_stop_valid, SILENT, compute_accuracy, plot_xx=None, plot_yy=None, plot_yy_noisy=None):
    print(" ~~~~~~~~~~Example 1 : Split 1~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=None,
                         early_stop_valid=None, nb_split=1, percent_validation_for_1_fold=20, compute_accuracy=False,
                         silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy, plot_yy_noisy=plot_yy_noisy,
                         testing_X=testing_X, testing_Y=testing_Y)
    if compute_accuracy:
        test_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=None,
                      early_stop_valid=None, nb_split=1, percent_validation_for_1_fold=20, compute_accuracy=True,
                      silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 2 : Split 1 with both stopper~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=early_stop_train,
                         early_stop_valid=early_stop_valid, nb_split=1, percent_validation_for_1_fold=20,
                         compute_accuracy=False, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                         plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y)
    if compute_accuracy:
        test_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=early_stop_train,
                      early_stop_valid=early_stop_valid, nb_split=1, percent_validation_for_1_fold=20,
                      compute_accuracy=True, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 3 : Split 5~~~~~~~~~~ ")
    if not compute_accuracy:
        print("todo implement.")

    if compute_accuracy:
        test_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=None,
                      early_stop_valid=None, nb_split=5, percent_validation_for_1_fold=0, compute_accuracy=True,
                      silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 3 : Split 5 with both stoppers~~~~~~~~~~ ")
    if not compute_accuracy:
        print("todo implement.")

    if compute_accuracy:
        test_accuracy(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train, early_stop_valid,
                      nb_split=5, percent_validation_for_1_fold=0, compute_accuracy=True, silent=SILENT,
                      testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 4 : no validation for 1 split ~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy_no_validation(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=None,
                                       early_stop_valid=None, nb_split=1, percent_validation_for_1_fold=0,
                                       compute_accuracy=False, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                                       plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y)
    if compute_accuracy:
        test_accuracy_no_validation(train_X, train_Y, parametrized_NN, parameters_training, early_stop_train=None,
                                    early_stop_valid=None, nb_split=1, percent_validation_for_1_fold=0,
                                    compute_accuracy=True, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    APlot.show_plot()
