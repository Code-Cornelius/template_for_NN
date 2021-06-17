from priv_lib_estimator import Evolution_plot_estimator
from priv_lib_plot import AColorsetContinuous

from nn_classes.estimator.plot_estim_history import Plot_estim_history


class Plot_evol_history(Plot_estim_history, Evolution_plot_estimator):
    EVOLUTION_COLUMN = 'epoch'

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)
        return

    def get_data2evolution(self, data, feature_to_draw):
        return self.get_data2group_sliced(data, feature_to_draw).mean().to_numpy()

    def get_default_dict_fig(self, grouped_data_by, key=None):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="")
        fig_dict = {'title': title,
                    'xlabel': self.EVOLUTION_COLUMN,
                    'ylabel': 'Loss',
                    'xscale': 'log', 'yscale': 'log',
                    'basex': 10, 'basey': 10
                    }
        return fig_dict

    def draw(self, second_axis = None, log_axis_for_loss = True,             log_second_axis = True):
        list_column_names = self.estimator.get_col_metric_names()[:4]

        # colormaps
        # adjusting the linewidth depending on nb of plots:
        if self.estimator.nb_folds < 3:
            linewidth = 2
            interval_colors = (0.5, 0.9)
        else:
            linewidth = 1
            interval_colors = (0.3, 1.)
        color_plot_blue = AColorsetContinuous('Blues', self.estimator.nb_folds, interval_colors)
        color_plot_green = AColorsetContinuous('Greens', self.estimator.nb_folds, interval_colors)
        color_plot_red = AColorsetContinuous('Reds', self.estimator.nb_folds, interval_colors)
        color_plot_orange = AColorsetContinuous('Oranges', self.estimator.nb_folds, interval_colors)
        colors = [color_plot_blue, color_plot_green, color_plot_red, color_plot_orange]
        for col_name,c in zip(list_column_names, colors):
            self.COLORMAP = c
            plots,_ = super().draw(column_name_draw = col_name, separator_colour='fold',
                                   dict_plot_for_main_line = {'linewidth':linewidth},
                                   true_values_flag = False, envelope_flag= False)
            plots = plots[0]


            if log_axis_for_loss:
                yscale = 'log'
            else:
                yscale = 'linear'

            # todo change plots yscale.

        return plots

        #
        # for i in range(self.estimator.nb_folds):
        #     dict_plot_param_loss_training = {"color": color_plot_green[i],
        #                                      "linewidth": linewidth,
        #                                      "label": f"Loss for Training nb {i + 1}"
        #                                      }
        #     aplot.uni_plot(nb_ax=0, xx=xx, yy=estimator_history.get_values_fold_col(i, 'loss_training'),
        #                    dict_plot_param=dict_plot_param_loss_training,
        #                    dict_ax={'title': "Training of a Neural Network, {}, evolution wrt epochs.".format(title),
        #                             'xlabel': "Epochs", 'ylabel': "Loss",
        #                             'xscale': 'linear', 'yscale': yscale,
        #                             'basey': 10})
        #     if key_for_second_axis_plot is not None:
        #         dict_plot_param_second_metric_training = {"color": color_plot_blue[i],
        #                                                   "linewidth": linewidth,
        #                                                   "label": f"{key_for_second_axis_plot} for Training nb {i + 1}"
        #                                                   }
        #         aplot.uni_plot_ax_bis(nb_ax=0, xx=xx,
        #                               yy=estimator_history.get_values_fold_col(i,
        #                                                                        f"{key_for_second_axis_plot}_training"),
        #                               dict_plot_param=dict_plot_param_second_metric_training,
        #                               dict_ax={'ylabel': key_for_second_axis_plot})
        #
        # _plot_validation_history(aplot, color_plot_orange, color_plot_red, flag_valid, estimator_history,
        #                          key_for_second_axis_plot,
        #                          linewidth, nb_trials, xx, log_axis_for_second_axis)
        #
        # # plot lines of best NN:
        # if len(estimator_history.best_epoch) > 0:
        #     _plot_best_epoch_NN(aplot, estimator_history.best_epoch, self.estimator.nb_folds)
        #
        # aplot.show_legend()
        # aplot._axs[0].grid(True)
        # if key_for_second_axis_plot is not None:
        #     aplot._axs_bis[0].grid(True)