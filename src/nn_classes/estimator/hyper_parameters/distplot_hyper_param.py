import seaborn as sns
from priv_lib_estimator import Distplot_estimator

from nn_classes.estimator.hyper_parameters.plot_estim_hyper_paramm import Plot_estim_hyper_param


class Distplot_hyper_param(Plot_estim_hyper_param, Distplot_estimator):
    def ___init__(self, estimator, *args, **kwargs):
        super().__init__(estimator=estimator, *args, **kwargs)

    def get_dict_fig(self, separators, key):
        title = self.generate_title(parameters=separators, parameters_value=key,
                                    before_text="Histogram for the hyper params")
        fig_dict = {'title': title,
                    'xlabel': "Loss",
                    'ylabel': "Nb of bins"}
        return fig_dict

