import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from IPython.core.display import display
from priv_lib_plot import APlot
from sklearn import metrics

import seaborn as sns


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


def nn_plot_prediction_vs_true(net, plot_xx, plot_yy=None, plot_yy_noisy=None):
    # todo add a title to the graph
    aplot = APlot(how=(1, 1))
    plot_yy_pred = net.nn_predict_ans2cpu(plot_xx)

    if plot_yy_noisy is not None:
        aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_noisy,
                       dict_plot_param={"color": "black",
                                        "linestyle": "--",
                                        "linewidth": 0.3,
                                        "markersize": 0,
                                        "label": "Noisy Trained over Solution"
                                        })
    if plot_yy is not None:
        aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy,
                       dict_plot_param={"color": "orange",
                                        "linewidth": 1,
                                        "label": "Solution"
                                        })

    aplot.uni_plot(nb_ax=0, xx=plot_xx, yy=plot_yy_pred,
                   dict_plot_param={"color": "c",
                                    "linewidth": 2,
                                    "label": "Predicted Data used for Training"
                                    }, dict_ax= {"xlabel": "Time", "ylabel": "Estimation",
                                    "title": "Visualization of prediction and true solution"})
    aplot.show_legend()
    return


def nn_print_errors(net, train_X, train_Y, testing_X, testing_Y):
    # Compute the relative validation error
    relative_error_train = torch.mean((net.nn_predict_ans2cpu(train_X) - train_Y) ** 2) / torch.mean(train_Y ** 2)
    print("Relative Training Error: ", relative_error_train.numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    # relative_error_val = torch.mean((net.nn_predict(validation_X) - validation_Y) ** 2) / torch.mean(validation_Y ** 2)
    # print("Relative Validation Error: ", relative_error_val.numpy() ** 0.5 * 100, "%")

    # Compute the relative L2 error norm (generalization error)
    relative_error_test = torch.mean((net.nn_predict_ans2cpu(testing_X) - testing_Y) ** 2) / torch.mean(testing_Y ** 2)
    print("Relative Testing Error: ", relative_error_test.numpy() ** 0.5 * 100, "%")
    return
