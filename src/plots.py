import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from sklearn.model_selection import GridSearchCV
import time
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.metrics

# for clustering
import sklearn.cluster

# for graphs:
import networkx as nx
import csv
from operator import itemgetter

# some statistics
import statistics
import seaborn as sns

sns.set()

def confusion_matrix_creator(Y, Y_predict_result, labels):
    confusion_matrix = sklearn.metrics.confusion_matrix(Y, Y_predict_result)  # we create a raw confusion matrix
    # confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(Y,Y_predict_result)
    cm_sum = np.sum(confusion_matrix, axis=1).reshape(-1,
                                                      1)  # we get the sum of the lines and reshape it (we re going to use it for the percentage)
    cm_percentage = confusion_matrix / cm_sum.astype(
        float) * 100  # we get a matrix of percentage. (row proportion for every column)
    annot = np.empty_like(confusion_matrix).astype(
        str)  # we create a raw array for the annotation that we will put on the final result

    n_rows, n_cols = confusion_matrix.shape  # getting the size of the matrix
    for i in range(0, n_rows):  # here that part is for getting the right annotation at its place.
        for j in range(0,
                       n_cols):  # the idea is that we want, the percentage, then the number that fits in it, and for diagonal elements, the sum of all the elements on the line.
            p = cm_percentage[i, j]
            c = confusion_matrix[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    fig, ax = plt.subplots(figsize=(15, 15))  # we set the frame
    sns.heatmap(confusion_matrix, annot=annot, fmt='', ax=ax, linewidths=.5, cmap="coolwarm")
    # using heatmap and setting some parameter for the confusion matrix.
    ax.set_xlabel('Predicted')  # here this line and the next is for putting the meaning of the cases
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')  # title
    ax.xaxis.set_ticklabels(labels)  # putting the meaning of each column and row
    ax.yaxis.set_ticklabels(labels)
    ax.set_ylim(n_rows + 0.1, -0.1)  # expanding the graph... without it, the squares are cut in the middle.
    ax.set_xlim(-0.1, n_cols + 0.1)
    plt.show()
    return


def result_function(title, data_train_Y, y_pred1, data_test_Y=None, y_pred2=None):
    # this function takes into input the results and return the scores.
    # I really simply made the function allowing either only data train or data train and test,
    # by copying and putting an if. The function is really simple so there is no need to do more complicate things.
    if data_test_Y is not None:
        conclusion = pd.DataFrame({'Set': ["Training", "Test"],
                                   'Accuracy': [sklearn.metrics.accuracy_score(data_train_Y, y_pred1),
                                                sklearn.metrics.accuracy_score(data_test_Y, y_pred2)]})
        display(conclusion)

        print("Confusion matrix for the train data set with " + title + ".")
        confusion_matrix_creator(data_train_Y, y_pred1, range(10))
        print("Confusion matrix for the test data set with " + title + ".")
        confusion_matrix_creator(data_test_Y, y_pred2, range(10))

        print("report table for train :")
        print(sklearn.metrics.classification_report(data_train_Y, y_pred1,
                                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
        print("report table for test :")
        print(sklearn.metrics.classification_report(data_test_Y, y_pred2,
                                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))

    else:
        conclusion = pd.DataFrame({'Set': ["Training"],
                                   'Accuracy': [sklearn.metrics.accuracy_score(data_train_Y, y_pred1)]})
        display(conclusion)
        print("Confusion matrix for the train data set with " + title + ".")
        confusion_matrix_creator(data_train_Y, y_pred1, range(10))

        print("report table for train :")
        print(sklearn.metrics.classification_report(data_train_Y, y_pred1,
                                                    target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
    return


# function for plotting the results of the NN
def nn_plot(mean_training_acc, mean_train_losses, mean_valid_acc=None, mean_valid_losses=None):
    fig = plt.figure(figsize=(15, 8))
    plt.grid(True)
    fig.tight_layout  # for the display
    ax = plt.axes()
    ax_bis = ax.twinx()
    ax.set_xlabel("Epochs")
    ax_bis.set_ylabel("Accuracy")
    ax.set_ylabel("Loss")

    x = range(len(mean_training_acc))
    y1 = mean_training_acc
    z1 = mean_train_losses
    alpha = ax_bis.plot(x, y1, 'bo-', markersize=3, label="Training Accuracy")
    gamma = ax.plot(x, z1, 'ro-', markersize=3, label="Training Loss")

    # lns is for the labels.
    lns = alpha + gamma
    # depending on if there is a validation set in the previous computations
    if mean_valid_acc is not None:
        y2 = mean_valid_acc
        z2 = mean_valid_losses
        beta = ax_bis.plot(x, y2, 'co-', markersize=3, label="Validation Accuracy")
        delta = ax.plot(x, z2, 'mo-', markersize=3, label="Validation Loss")
        lns += beta + delta

    # here we set the legend as it has to be.
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="center right")
    plt.suptitle("Training of a Neural Network, presentation of the evolution of the accuracy and of the loss", y=0.94,
                 fontsize=20)
    return