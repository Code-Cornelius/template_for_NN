import os

import pandas as pd
from priv_lib_estimator import Estimator

from nn_classes.architecture.fully_connected import Fully_connected_NN
from nn_classes.estimator.history.estim_history import Estim_history


class Estim_hyper_param(Estimator):

    def __init__(self, df=None):
        super().__init__(df)

    @classmethod
    def from_folder(cls, path, metric_name, compressed=True):
        """
        Semantics:
            Initialise an estim_hyper_param from a folder of estim_history.
        Args:
            path: The path to the folder.
            metric_name: The metric used for comparison.

        Returns:
            An Estim_hyper_param.
        """
        estimators = Estim_history.folder_json2list_estim(path, compressed)

        return Estim_hyper_param.from_list(estimators, metric_name)

    @classmethod
    def from_list(cls, estimators, metric_name):
        """
        Semantics:
            Initialise an estim_hyper_param from a list of estim_history.
        Args:
            estimators(list of Estim_history): The estimators to be used.
            metric_name: The metric used for comparison.

        Returns:
            An Estim_hyper_param.
        """
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

    def compute_number_params_for_fcnn(self):
        """
        Semantics:
            Computes the number of parameters for each entry and adds it to a new column.
        Requirements:
            Works for a fully connected NN and the estimator must contain the input_sizes, the list of hidden sizes
            and the output sizes.
        Returns:
            Void.
        """
        assert 'input_size' in self.df.columns, "Cannot compute the number of parameters without input_size"
        assert 'list_hidden_sizes' in self.df.columns, "Cannot compute the number of parameters without" \
                                                       " list_hidden_sizes"
        assert 'output_size' in self.df.columns, "Cannot compute the number of parameters without output_size"

        self.df['list_hidden_sizes'] = pd.eval(self.df['list_hidden_sizes'])

        self.df['nb_of_params'] = self.df.apply(
            lambda row: Fully_connected_NN.compute_nb_of_params(input_size=row.input_size,
                                                                list_hidden_sizes=row.list_hidden_sizes,
                                                                output_size=row.output_size),
            axis=1
        )


