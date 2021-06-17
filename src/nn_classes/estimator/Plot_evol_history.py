import numpy as np
from priv_lib_estimator import Evolution_plot_estimator
from priv_lib_plot import AColorsetContinuous, APlot

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
                    'basex': 10, 'basey': 10}
        return fig_dict

    def draw(self, key_for_second_axis_plot=None, log_axis_for_loss=True, log_axis_for_second_axis=False):
        # TODO CHECK THE KEY FOR SECOND AXIS IS INDEED IN THE COLUMNS
        # TODO list for best_epoch in case 1 element
        aplot = APlot()

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

        if log_axis_for_loss:
            yscale = 'log'
        else:
            yscale = 'linear'

        coloured_dict, coloured_keys = self.estimator.groupby('fold')
        for i, coloured_key in enumerate(coloured_keys):  # iterating over the folds.
            coloured_data = coloured_dict.get_group(coloured_key)
            evolution_xx = self.get_values_evolution_column(coloured_data)

            dict_plot_param_loss_training = {"color": color_plot_green[i],
                                             "linewidth": linewidth,
                                             "label": f"Loss for Training nb {i + 1}"}

            aplot.uni_plot(nb_ax=0, xx=evolution_xx, yy=self.get_data2evolution(coloured_data, 'loss_training'),
                           dict_plot_param=dict_plot_param_loss_training,
                           dict_ax={'title': "Training of a Neural Network, evolution wrt epochs.",
                                    'xlabel': "Epochs", 'ylabel': "Loss",
                                    'xscale': 'linear', 'yscale': yscale,
                                    'basey': 10})
            if key_for_second_axis_plot is not None:
                dict_plot_param_second_metric_training = {"color": color_plot_blue[i],
                                                          "linewidth": linewidth,
                                                          "label": f"{key_for_second_axis_plot} for Training nb {i + 1}"
                                                          }
                aplot.uni_plot_ax_bis(nb_ax=0, xx=evolution_xx,
                                      yy=self.get_data2evolution(coloured_data, f"{key_for_second_axis_plot}_training"),
                                      dict_plot_param=dict_plot_param_second_metric_training,
                                      dict_ax={'ylabel': key_for_second_axis_plot})

        flag_valid = self.estimator.validation

        if flag_valid:
            if log_axis_for_second_axis:
                dict_ax = {'yscale': 'log'}
            else:
                dict_ax = None

            for i, coloured_key in enumerate(coloured_keys):  # iterating over the folds.
                coloured_data = coloured_dict.get_group(coloured_key)
                evolution_xx = self.get_values_evolution_column(coloured_data)
                dict_plot_param_loss_validation = {"color": color_plot_orange[i],
                                                   "linewidth": linewidth,
                                                   "label": f"Loss for Validation nb {i + 1}"
                                                   }
                aplot.uni_plot(nb_ax=0, xx=evolution_xx, yy=self.get_data2evolution(coloured_data, 'loss_validation'),
                               dict_plot_param=dict_plot_param_loss_validation)
                if key_for_second_axis_plot is not None:
                    dict_plot_param_second_metric_validation = {"color": color_plot_red[i],
                                                                "linewidth": linewidth,
                                                                "label": f"{key_for_second_axis_plot} for Validation nb {i + 1}"
                                                                }
                    aplot.uni_plot_ax_bis(nb_ax=0, xx=evolution_xx,
                                          yy=self.get_data2evolution(coloured_data,
                                                                     f"{key_for_second_axis_plot}_validation"),
                                          dict_plot_param=dict_plot_param_second_metric_validation, dict_ax=dict_ax)

        # plot lines of best NN:
        # TODO LIST
        if len(list(self.estimator.best_epoch)) > 0:
            # TODO LIST
            _plot_best_epoch_NN(aplot, list(self.estimator.best_epoch))

        aplot.show_legend()
        aplot._axs[0].grid(True)
        if key_for_second_axis_plot is not None:
            aplot._axs_bis[0].grid(True)

        return aplot


def _plot_best_epoch_NN(aplot, best_epoch_of_NN):
    yy = np.array(aplot.get_y_lim(nb_ax=0))
    for i in range(len(best_epoch_of_NN)):
        aplot.plot_vertical_line(best_epoch_of_NN[i], yy, nb_ax=0,
                                 dict_plot_param={"color": "black",
                                                  "linestyle": "--",
                                                  "linewidth": 0.3,
                                                  "markersize": 0,
                                                  "label": f"Best model for fold nb {i + 1}"
                                                  })
