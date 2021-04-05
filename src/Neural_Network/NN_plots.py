import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from IPython.core.display import display
from priv_lib_plot import APlot
from sklearn import metrics

import seaborn as sns

from src.Neural_Network.NN_fcts import nn_predict

sns.set()


def confusion_matrix_creator(Y, Y_predict_result, labels, title=""):
    # we create a raw confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y, Y_predict_result)

    # we get the sum of the lines and reshape it (we re going to use it for the percentage)
    cm_sum = np.sum(confusion_matrix, axis=1).reshape(-1, 1)

    # we get a matrix of percentage. (row proportion for every column)
    cm_percentage = confusion_matrix / cm_sum.astype(float) * 100

    # we create a raw array for the annotation that we will put on the final result
    annot = np.empty_like(confusion_matrix).astype(str)

    # getting the size of the matrix
    n_rows, n_cols = confusion_matrix.shape

    # here that part is for getting the right annotation at its place.
    for i in range(0, n_rows):
        # the idea is that we want, the percentage, then the number that fits in it,
        # and for diagonal elements, the sum of all the elements on the line.
        for j in range(0, n_cols):
            p = cm_percentage[i, j]
            c = confusion_matrix[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    # we set the frame
    fig, ax = plt.subplots()

    # using heatmap and setting some parameter for the confusion matrix.
    sns.set(font_scale=0.6)
    sns.heatmap(confusion_matrix, annot=annot, fmt='', ax=ax, linewidths=.5, cmap="coolwarm")
    sns.set(font_scale=1)

    # here this line and the next is for putting the meaning of the cases
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix {title}")  # title
    ax.xaxis.set_ticklabels(labels)  # putting the meaning of each column and row
    ax.yaxis.set_ticklabels(labels)
    ax.set_ylim(n_rows + 0.1, -0.1)  # expanding the graph... without it, the squares are cut in the middle.
    ax.set_xlim(-0.1, n_cols + 0.1)


def result_function(title, data_train_Y, y_pred1, no_classes, data_test_Y=None, y_pred2=None):
    # initialise the parameters for the dataframe
    conclusion_set = ["Training"]
    conclusion_accuracy = [metrics.accuracy_score(data_train_Y, y_pred1)]

    if data_test_Y is not None:
        conclusion_set.append("Test")
        conclusion_accuracy.append(metrics.accuracy_score(data_test_Y, y_pred2))

    conclusion = pd.DataFrame({"Set": conclusion_set,
                               "Accuracy": conclusion_accuracy})
    display(conclusion)

    print("Confusion matrix for the train data set with " + title + ".")
    confusion_matrix_creator(data_train_Y, y_pred1, range(no_classes))

    if data_test_Y is not None:
        print("Confusion matrix for the test data set with " + title + ".")
        confusion_matrix_creator(data_test_Y, y_pred2, range(no_classes))

    target_names = [str(i) for i in range(no_classes)]
    print("report table for train :")
    print(metrics.classification_report(data_train_Y, y_pred1, target_names=target_names))

    if data_test_Y is not None:
        print("report table for test :")
        print(metrics.classification_report(data_test_Y, y_pred2, target_names=target_names))


def nn_plot_train_loss_acc(training_loss, validation_loss=None, training_acc=None, validation_acc=None,
                           log_axis_for_loss=True, best_epoch_of_NN=None):
    if training_acc is not None:
        aplot = APlot(how=(1, 1), sharex=True)
    else:
        aplot = APlot(how=(1, 1), sharex=False)
    nb_of_epoch = training_loss.shape[1]
    nb_trials = training_loss.shape[0]
    xx = range(nb_of_epoch)

    Blues = plt.get_cmap('Blues')
    color_plot_blue = Blues(np.linspace(0.3, 1, nb_trials))  # color map for plot

    Greens = plt.get_cmap('Greens')
    color_plot_green = Greens(np.linspace(0.3, 1, nb_trials))  # color map for plot

    Reds = plt.get_cmap('Reds')
    color_plot_red = Reds(np.linspace(0.3, 1, nb_trials))  # color map for plot

    Oranges = plt.get_cmap('Oranges')
    color_plot_orange = Oranges(np.linspace(0.3, 1, nb_trials))  # color map for plot

    if log_axis_for_loss:
        yscale = "log"
    else:
        yscale = "linear"
    # adjusting the linewidth depending on nb of plots:
    if nb_trials < 3:
        linewidth = 2
    else:
        linewidth = 1

    for i in range(nb_trials):
        dict_plot_param_loss_training = {"color": color_plot_green[i],
                                         "linewidth": linewidth,
                                         "label": f"Loss for Training nb {i}"
                                         }
        aplot.uni_plot(nb_ax=0, xx=xx, yy=training_loss[i, :],
                       dict_plot_param=dict_plot_param_loss_training,
                       dict_ax={'title': "Training of a Neural Network, evolution wrt epochs.",
                                'xlabel': "Epochs", 'ylabel': "Loss",
                                'xscale': 'linear', 'yscale': yscale,
                                'basey': 10})
        if training_acc is not None:
            dict_plot_param_accuracy_training = {"color": color_plot_blue[i],
                                                 "linewidth": linewidth,
                                                 "label": f"Accuracy for Training nb {i}"
                                                 }
            aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=training_acc[i, :],
                                  dict_plot_param=dict_plot_param_accuracy_training)

    if validation_loss is not None:
        _plot_validation_history(aplot, color_plot_orange, color_plot_red, linewidth, nb_trials, validation_acc,
                                 validation_loss, xx)
    # plot lines of best NN:
    if best_epoch_of_NN is not None:
        _plot_best_epoch_NN(aplot, best_epoch_of_NN, nb_trials)

    aplot.show_legend()
    aplot._axs[0].grid(True)
    if training_acc is not None:
        aplot._axs_bis[0].grid(True)

    return


def _plot_best_epoch_NN(aplot, best_epoch_of_NN, nb_trials):
    yy = np.array(aplot.get_y_lim(nb_ax=0))
    for i in range(nb_trials):
        aplot.plot_vertical_line(best_epoch_of_NN[i], yy, nb_ax=0, dict_plot_param={"color": "black",
                                                                                "linestyle": "--",
                                                                                "linewidth": 0.3,
                                                                                "markersize": 0,
                                                                                "label": f"Best model for fold nb {i}"
                                                                                })


def _plot_validation_history(aplot, color_plot_orange, color_plot_red, linewidth, nb_trials, validation_acc,
                             validation_loss, xx):
    for i in range(nb_trials):
        dict_plot_param_loss_validation = {"color": color_plot_orange[i],
                                           "linewidth": linewidth,
                                           "label": f"Loss for Validation nb {i}"
                                           }
        aplot.uni_plot(nb_ax=0, xx=xx, yy=validation_loss[i, :], dict_plot_param=dict_plot_param_loss_validation)
        if validation_acc is not None:
            dict_plot_param_accuracy_validation = {"color": color_plot_red[i],
                                                   "linewidth": linewidth,
                                                   "label": f"Accuracy for Validation nb {i}"
                                                   }
            aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=validation_acc[i, :],
                                  dict_plot_param=dict_plot_param_accuracy_validation)


def nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy):
    aplot = APlot(how=(1, 1))
    plot_yy_pred = nn_predict(net, plot_xx)

    aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_noisy, dict_plot_param={"color": "black",
                                                                           "linestyle": "--",
                                                                           "linewidth": 0.3,
                                                                           "markersize": 0,
                                                                           "label": "Noisy Trained over Solution"
                                                                           })

    aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy, dict_plot_param={"color": "orange",
                                                                     "linewidth": 1,
                                                                     "label": "Solution"
                                                                     })

    aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_pred, dict_plot_param={"color": "c",
                                                                          "linewidth": 2,
                                                                          "label": "Predicted Data used for Training"
                                                                          })
    aplot.show_legend()
    return


def nn_print_errors(net, train_X, train_Y, testing_X, testing_Y):
    # Compute the relative validation error
    relative_error_train = torch.mean((nn_predict(net, train_X) - train_Y) ** 2) / torch.mean(train_Y ** 2)
    print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    # relative_error_val = torch.mean((nn_predict(net, validation_X) - validation_Y) ** 2) / torch.mean(validation_Y ** 2)
    # print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative L2 error norm (generalization error)
    relative_error_test = torch.mean((nn_predict(net, testing_X) - testing_Y) ** 2) / torch.mean(testing_Y ** 2)
    print("Relative Testing Error: ", relative_error_test.detach().numpy() ** 0.5 * 100, "%")
    return
