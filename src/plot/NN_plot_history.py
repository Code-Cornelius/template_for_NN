import numpy as np

# my lib
from priv_lib_plot import APlot
from priv_lib_plot import AColorsetContinuous


def nn_plot_train_loss_acc(
        history, key_for_second_axis_plot=None, flag_valid=True,
        log_axis_for_loss=True, best_epoch_of_NN=None,
        log_axis_for_second_axis=False, title=''):
    # if there is another key, we create a plot with two y-axis but only one x-axis.
    if key_for_second_axis_plot is not None:
        aplot = APlot(how=(1, 1), sharex=True)
    else:
        aplot = APlot(how=(1, 1), sharex=False)

    training_loss = history['training']['loss']

    xx = range(training_loss.shape[1])
    nb_trials = training_loss.shape[0]

    if log_axis_for_loss:
        yscale = 'log'
    else:
        yscale = 'linear'
    # adjusting the linewidth depending on nb of plots:
    if nb_trials < 3:
        linewidth = 2
        interval_colors = (0.5, 0.9)
    else:
        linewidth = 1
        interval_colors = (0.3, 1.)

    color_plot_blue = AColorsetContinuous('Blues', nb_trials, interval_colors)
    color_plot_green = AColorsetContinuous('Greens', nb_trials, interval_colors)
    color_plot_red = AColorsetContinuous('Reds', nb_trials, interval_colors)
    color_plot_orange = AColorsetContinuous('Oranges', nb_trials, interval_colors)

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
        if key_for_second_axis_plot is not None:
            dict_plot_param_accuracy_training = {"color": color_plot_blue[i],
                                                 "linewidth": linewidth,
                                                 "label": f"{key_for_second_axis_plot} for Training nb {i}"
                                                 }
            aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=history['training'][key_for_second_axis_plot][i, :],
                                  dict_plot_param=dict_plot_param_accuracy_training,
                                  dict_ax={'ylabel': key_for_second_axis_plot})

    _plot_validation_history(aplot, color_plot_orange, color_plot_red, flag_valid, history, key_for_second_axis_plot,
                             linewidth, nb_trials, xx, log_axis_for_second_axis)

    # plot lines of best NN:
    if best_epoch_of_NN is not None:
        _plot_best_epoch_NN(aplot, best_epoch_of_NN, nb_trials)

    aplot.show_legend()
    aplot._axs[0].grid(True)
    if key_for_second_axis_plot is not None:
        aplot._axs_bis[0].grid(True)

    return


def _plot_validation_history(aplot, color_plot_loss_validation, color_plot_red, flag_valid, history,
                             key_for_second_axis_plot, linewidth, nb_trials, xx, log_axis_for_second_axis):
    if flag_valid:
        if log_axis_for_second_axis:
            dict_ax = {'yscale': 'log'}
        else:
            dict_ax = None

        for i in range(nb_trials):
            dict_plot_param_loss_validation = {"color": color_plot_loss_validation[i],
                                               "linewidth": linewidth,
                                               "label": f"Loss for Validation nb {i}"
                                               }
            aplot.uni_plot(nb_ax=0, xx=xx, yy=history['validation']['loss'][i, :],
                           dict_plot_param=dict_plot_param_loss_validation)
            if key_for_second_axis_plot is not None:
                dict_plot_param_accuracy_validation = {"color": color_plot_red[i],
                                                       "linewidth": linewidth,
                                                       "label": f"{key_for_second_axis_plot} for Validation nb {i}"
                                                       }
                aplot.uni_plot_ax_bis(nb_ax=0, xx=xx, yy=history['validation'][key_for_second_axis_plot][i, :],
                                      dict_plot_param=dict_plot_param_accuracy_validation,
                                      dict_ax=dict_ax)


def _plot_best_epoch_NN(aplot, best_epoch_of_NN, nb_trials):
    yy = np.array(aplot.get_y_lim(nb_ax=0))
    for i in range(nb_trials):
        aplot.plot_vertical_line(best_epoch_of_NN[i], yy, nb_ax=0,
                                 dict_plot_param={"color": "black",
                                                  "linestyle": "--",
                                                  "linewidth": 0.3,
                                                  "markersize": 0,
                                                  "label": f"Best model for fold nb {i}"
                                                  })
