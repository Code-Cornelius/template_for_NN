import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.display import display
from priv_lib_plot import APlot
from sklearn import metrics

import seaborn as sns

sns.set()


def confusion_matrix_creator(Y, Y_predict_result, labels, title = ""):
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


def nn_plot(training_loss, validation_loss=None, training_acc=None, validation_acc=None, log_axis_for_loss=True):
    if training_acc is not None:
        aplot = APlot(how=(1, 1), sharex=True)
    else:
        aplot = APlot(how=(1, 1), sharex=False)
    nb_of_epoch = training_acc.shape[1]
    nb_trials = training_acc.shape[0]
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

    for i in range(nb_trials):
        dict_plot_param_loss_training = {"color": color_plot_green[i],
                                         "linewidth": 1,
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
                                                 "linewidth": 1,
                                                 "label": f"Accuracy for Training nb {i}"
                                                 }
            aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=training_acc[i, :],
                                  dict_plot_param=dict_plot_param_accuracy_training)

    if validation_loss is not None:
        for i in range(nb_trials):
            dict_plot_param_loss_validation = {"color": color_plot_orange[i],
                                               "linewidth": 1,
                                               "label": f"Loss for Validation nb {i}"
                                               }
            aplot.uni_plot(nb_ax=0, xx=xx, yy=validation_loss[i, :], dict_plot_param=dict_plot_param_loss_validation)
            if validation_acc is not None:
                dict_plot_param_accuracy_validation = {"color": color_plot_red[i],
                                                       "linewidth": 1,
                                                       "label": f"Accuracy for Validation nb {i}"
                                                       }
                aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=validation_acc[i, :],
                                      dict_plot_param=dict_plot_param_accuracy_validation)

    aplot.show_legend()
    aplot._axs[0].grid(True)
    aplot._axs_bis[0].grid(True)
