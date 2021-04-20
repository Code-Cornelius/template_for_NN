from src.plot.NN_plot_history import nn_plot_train_loss_acc
from src.train.NN_kfold_training import nn_kfold_train
from src.plot.NN_plots import *
from src.training_stopper.Early_stopper_vanilla import Early_stopper_vanilla


def test_no_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split,
                     percent_validation_for_1_fold, silent, plot_xx, plot_yy, plot_yy_noisy, testing_X, testing_Y,
                     early_stoppers=[Early_stopper_vanilla()]):
    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN,
                                                      parameters_training=parameters_training,
                                                      early_stoppers=early_stoppers, nb_split=nb_split,
                                                      shuffle_kfold=True,
                                                      percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                      silent=silent)
    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(training_loss=history['training']['loss'], validation_loss=history['validation']['loss'],
                           best_epoch_of_NN=best_epoch_of_NN)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)


def test_no_accuracy_no_validation(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split,
                                   percent_validation_for_1_fold, silent, plot_xx, plot_yy, plot_yy_noisy, testing_X,
                                   testing_Y, early_stoppers=[Early_stopper_vanilla()]):
    print("coucou problem")
    print( early_stoppers[0].is_validation())
    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN,
                                                      parameters_training=parameters_training,
                                                      early_stoppers=early_stoppers, nb_split=nb_split,
                                                      shuffle_kfold=True,
                                                      percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                      silent=silent)
    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(training_loss=history['training']['loss'], validation_loss=None, best_epoch_of_NN=best_epoch_of_NN)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)


def test_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split, percent_validation_for_1_fold,
                  silent, testing_X, testing_Y, early_stoppers=[Early_stopper_vanilla()]):
    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN,
                                                      parameters_training=parameters_training,
                                                      early_stoppers=early_stoppers, nb_split=nb_split,
                                                      shuffle_kfold=True,
                                                      percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                      silent=silent)
    net.to(torch.device('cpu'))
    nn_plot_train_loss_acc(history['training']['loss'], history['validation']['loss'], history['training']["accuracy"],
                           history['validation']["accuracy"], best_epoch_of_NN=best_epoch_of_NN)
    # confusion_matrix_creator(train_Y, net.nn_predict(train_X), range(10), title="Training Set")
    # confusion_matrix_creator(testing_Y, net.nn_predict(testing_X), range(10), title="Test Set")


def test_accuracy_no_validation(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split,
                                percent_validation_for_1_fold, silent, testing_X, testing_Y,
                                early_stoppers=[Early_stopper_vanilla()]):
    (net, history, best_epoch_of_NN) = nn_kfold_train(train_X, train_Y, Class_Parametrized_NN,
                                                      parameters_training=parameters_training,
                                                      early_stoppers=early_stoppers, nb_split=nb_split,
                                                      shuffle_kfold=True,
                                                      percent_validation_for_1_fold=percent_validation_for_1_fold,
                                                      silent=silent)
    net.to(torch.device('cpu'))

    nn_plot_train_loss_acc(history['training']['loss'], None, history['training']['accuracy'], None, best_epoch_of_NN=best_epoch_of_NN)
    # confusion_matrix_creator(train_Y, net.nn_predict(train_X), range(10), title="Training Set")
    # confusion_matrix_creator(testing_Y, net.nn_predict(testing_X), range(10), title="Test Set")


def test(train_X, train_Y, Class_Parametrized_NN, parameters_training, testing_X, testing_Y, early_stoppers,
         SILENT, compute_accuracy, plot_xx=None, plot_yy=None, plot_yy_noisy=None):
    print(" ~~~~~~~~~~Example 1 : Split 1~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                         percent_validation_for_1_fold=20, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                         plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y)
    if compute_accuracy:
        test_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                      percent_validation_for_1_fold=20, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 2 : Split 1 with both stopper~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                         percent_validation_for_1_fold=20, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                         plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y,
                         early_stoppers=early_stoppers)
    if compute_accuracy:
        test_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                      percent_validation_for_1_fold=20, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y,
                      early_stoppers=early_stoppers)

    print(" ~~~~~~~~~~Example 3 : Split 3~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=3,
                         percent_validation_for_1_fold=0, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                         plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y)

    if compute_accuracy:
        test_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=3,
                      percent_validation_for_1_fold=0, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y)

    print(" ~~~~~~~~~~Example 3 : Split 5 with both stoppers~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=5,
                         percent_validation_for_1_fold=0, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                         plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y,
                         early_stoppers=early_stoppers)

    if compute_accuracy:
        test_accuracy(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=5,
                      percent_validation_for_1_fold=0, silent=SILENT, testing_X=testing_X, testing_Y=testing_Y,
                      early_stoppers=early_stoppers)

    print(" ~~~~~~~~~~Example 4 : no validation for 1 split ~~~~~~~~~~ ")
    if not compute_accuracy:
        test_no_accuracy_no_validation(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                                       percent_validation_for_1_fold=0, silent=SILENT, plot_xx=plot_xx, plot_yy=plot_yy,
                                       plot_yy_noisy=plot_yy_noisy, testing_X=testing_X, testing_Y=testing_Y)
    if compute_accuracy:
        test_accuracy_no_validation(train_X, train_Y, Class_Parametrized_NN, parameters_training, nb_split=1,
                                    percent_validation_for_1_fold=0, silent=SILENT, testing_X=testing_X,
                                    testing_Y=testing_Y)

    APlot.show_plot()
