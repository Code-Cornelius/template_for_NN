# normal libraries


# priv libraries
from priv_lib_error import Error_type_setter
from priv_lib_estimator import Plot_estimator, Evolution_plot_estimator
from priv_lib_estimator.src.estimator.estim_time import Estim_time


# section ######################################################################
#  #############################################################################
# Classes

class Estim_benchmark_perf_nn_sizes(Estim_time):
    CORE_COL = Estim_time.CORE_COL.copy()
    CORE_COL.update(("Input Size", "Processing Unit", "Model Size", "Depth"))  # add to the name_columns the specific columns.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Plot_estim_benchmark_perf_nn_sizes(Plot_estimator):
    def __init__(self, estimator_bench, *args, **kwargs):
        if not isinstance(estimator_bench, Estim_benchmark_perf_nn_sizes):
            raise Error_type_setter(f'Argument is not an {str(Estim_benchmark_perf_nn_sizes)}.')
        super().__init__(estimator_bench, *args, **kwargs)


class Plot_evol_benchmark_perf_nn_sizes(Plot_estim_benchmark_perf_nn_sizes, Evolution_plot_estimator):
    EVOLUTION_COLUMN = "Input Size"

    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)
        return

    def get_data2evolution(self, data, feature_to_draw):
        return self.get_data2group_sliced(data, feature_to_draw).mean().to_numpy()

    def get_default_dict_fig(self, grouped_data_by, key=None):
        title = self.generate_title(parameters=grouped_data_by, parameters_value=key,
                                    before_text="Benchmark of the average time per iteration of the model")
        fig_dict = {'title': title,
                    'xlabel': self.EVOLUTION_COLUMN,
                    'ylabel': 'Comput. Time',
                    'xscale': 'log', 'yscale': 'log',
                    'basex': 10, 'basey': 10
                    }
        return fig_dict