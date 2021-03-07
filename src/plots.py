import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.display import display
from sklearn import metrics

import seaborn as sns

sns.set()


def confusion_matrix_creator(Y, Y_predict_result, labels):
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
    sns.heatmap(confusion_matrix, annot=annot, fmt='', ax=ax, linewidths=.5, cmap="coolwarm")

    # here this line and the next is for putting the meaning of the cases
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')  # title
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

    conclusion = pd.DataFrame({'Set': conclusion_set,
                               'Accuracy': conclusion_accuracy})
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


# function for plotting the results of the NN
def nn_plot(mean_training_acc, mean_train_losses, mean_valid_acc=None, mean_valid_losses=None, log_axis_for_loss = True):
    fig = plt.figure()
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
    if log_axis_for_loss:
        ax.set_yscale('log')
    plt.suptitle("Training of a Neural Network, presentation of the evolution of the accuracy and of the loss", y=0.94,
                 fontsize=20)
