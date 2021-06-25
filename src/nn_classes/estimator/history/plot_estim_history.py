from priv_lib_error import Error_type_setter
from priv_lib_estimator import Plot_estimator

from nn_classes.estimator.history.estim_history import Estim_history


class Plot_estim_history(Plot_estimator):

    def __init__(self, estimator_bench, *args, **kwargs):
        if not isinstance(estimator_bench, Estim_history):
            raise Error_type_setter(f'Argument is not an {str(Estim_history)}.')
        super().__init__(estimator_bench, *args, **kwargs)


