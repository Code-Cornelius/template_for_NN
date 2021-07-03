import os

import pandas as pd
from priv_lib_estimator import Estimator

from nn_classes.estimator.history.estim_history import Estim_history


class Estim_hyper_param(Estimator):

    def __init__(self, df=None):
        super().__init__(df)

    @classmethod
    def from_folder(cls, path, metric_name, compressed=True):
        """
            Initialise an estim_training_parameters from a folder of estim_history
        Args:
            path: The path to the folder
            metric_name: The metric used for comparison

        Returns:

        """
        estimators = Estim_history.folder2list_estim(path, compressed)

        # collect the data from the estimators
        dataframe_information = [Estim_hyper_param._get_dict_from_estimator(estimator, metric_name)
                                 for estimator in estimators]

        # initialise the dataframe
        dataframe = pd.DataFrame(dataframe_information)
        return cls(dataframe)

    @staticmethod
    def _get_dict_from_estimator(estimator, metric_name):
        estimator_dict = estimator.hyper_params.copy()
        estimator_dict[metric_name] = estimator.get_best_value_for(metric_name)

        return estimator_dict
