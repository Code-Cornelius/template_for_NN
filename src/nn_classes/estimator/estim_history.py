import json

import numpy as np
import pandas as pd
import numpy as np
from priv_lib_error import Error_type_setter
from priv_lib_estimator import Estimator
from priv_lib_util.tools.src.function_json import unzip_json, zip_json


class Estim_history(Estimator):
    NAMES_COLUMNS = {'fold', 'epoch'}

    def __init__(self, metric_names=None, validation=True, hyper_params=None):
        # not possible anymore to init with a df.
        self.metric_names = metric_names
        self.validation = validation
        self.best_epoch = []
        self.hyper_params = hyper_params
        self.best_fold = -1

        df_column_names = self._generate_all_column_names()
        super().__init__(pd.DataFrame(columns=df_column_names))

    # section ######################################################################
    #  #############################################################################
    #  JSON constructor and saver.

    def to_json(self, path, compress=True, *kwargs):
        """
            Save an estimator to json as a compressed file.
        Args:
            compress: whether or not compression is applied
            path: The path where to store the estimator

        Returns:`
            Void
        """
        attrs = {
            'validation': self.validation,
            'best_epoch': self.best_epoch,
            'hyper_params': self.hyper_params,
            'best_fold': self.best_fold
        }

        super().to_json(path, compress, attrs)

    @classmethod
    def from_json(cls, path, compressed=True):
        """
            Create estimator from previously stored json file
        Args:
            compressed: whether or not compression is applied
            path: The source path for the json

        Returns:
            Void
        """
        estimator = super().from_json(path)
        attrs = super().from_json_attributes(path, compressed)

        estimator.validation = attrs['validation']
        estimator.best_epoch = attrs['best_epoch']
        estimator.hyper_params = attrs['hyper_params']
        estimator.best_fold = attrs['best_fold']
        return estimator

    def get_col_metric_names(self):
        """
        Semantics:
            Generate the names for the columns which hold information about the metrics

        Returns:
            A list of strings representing the column names based on metric names and validation flag
        """
        df_column_names = []
        df_column_names.append(Estim_history._generate_column_name("loss"))

        for metric_name in self.metric_names:
            df_column_names.append(Estim_history._generate_column_name(metric_name))

        if self.validation:
            df_column_names.append(Estim_history._generate_column_name("loss", validation=True))

            for metric_name in self.metric_names:
                df_column_names.append(Estim_history._generate_column_name(metric_name, validation=True))

        return df_column_names


    @staticmethod
    def _generate_column_name(base_name, validation=False):
        """
        Semantics:
            Generate the column name based on a metric name and its use case (training or validation)

        Args:
            base_name: the name of the metric
            validation: boolean representing whether the metric is a result of validation or training

        Returns:
            The formatted column name
        """
        return f"{base_name}_" + ('validation' if validation else 'training')

    def _generate_all_column_names(self):
        """
        Generate all the column names for the dataframe
        Returns:
            A list of all the column names for the dataframe (Including the base columns)
        """
        df_column_names = list(Estim_history.NAMES_COLUMNS.copy()) + self.get_col_metric_names()

        return df_column_names

    def append_history(self, history, fold_best_epoch, fold_number):
        """
            Append information from history to the estimator
        Args:
            fold_best_epoch:
            history: history of the training
            fold_number: the fold number the history corresponds to

        Returns:
            Void
        """
        self.best_epoch.append(fold_best_epoch)
        adapted_history = self._translate_history_to_dataframe(history, fold_number)
        adapted_history = pd.DataFrame(adapted_history)
        self.append(adapted_history)

    def _translate_history_to_dataframe(self, history, fold_number):
        """
            Translate from history structure to a flat structure that will be used to add the history to the dataframe
        Args:
            history: the history of the training
            fold_number: the fold number the history corresponds to

        Returns:
            The translated history
        """
        translated_history = {}


        # collect training information
        for key, value in history['training'].items():
            new_key = self._generate_column_name(key)
            new_value = value[~np.isnan(value)]
            translated_history[new_key] = new_value.tolist()

        assert ('validation' in history) == self.validation, "The information about validation in estimator " \
                                                             "is not reflected in history"
        # collect validation information if present
        if 'validation' in history:
            for key, value in history['validation'].items():
                new_key = self._generate_column_name(key, validation=True)
                new_value = value[~np.isnan(value)]
                translated_history[new_key] = new_value.tolist()

        # add the epoch number to the translated history
        nb_epochs = len(translated_history['loss_training'])
        translated_history['epoch'] = [*range(nb_epochs)]

        # add the fold number to the history
        translated_history['fold'] = [fold_number] * nb_epochs

        return translated_history

    def _index_mask(self, fold, epoch):
        return (self.df.loc[:, 'fold'] == fold) & (self.df.loc[:, 'epoch'] == epoch)

    def _fold_mask(self, fold):
        return self.df.loc[:, 'fold'] == fold

    @property
    def nb_folds(self):
        return self.df.loc[:, 'fold'].max() + 1

    @property
    def nb_epochs(self):
        return self.df.loc[:, 'epoch'].max() + 1

    def slice_best_fold(self):
        """
        Semantics:
            Slice the dataframe to only store the best fold

        Returns:
            Void
        """
        self.best_epoch = [self.best_epoch[self.best_fold]]
        self.df = self.df.loc[self._fold_mask(self.best_fold)]

    # section ######################################################################
    #  #############################################################################
    # SETTERS GETTERS

    def get_values_fold_epoch_col(self, fold, epoch, column):
        index = self._index_mask(fold, epoch)
        return self.df.loc[:, column][index].values[0]

    def get_values_fold_col(self, fold, column):
        index = self._fold_mask(fold)
        return self.df.loc[:, column][index].values

    def get_values_col(self, column):
        return self.df.loc[:, column].values

    def get_best_value_for(self, column):
        if len(self.best_epoch) > 1:
            epoch = self.best_epoch[self.best_fold]
        else:
            epoch = self.best_epoch[0]

        return self.get_values_fold_epoch_col(self.best_fold, epoch, column)

    @property
    def best_fold(self):
        return self._best_fold

    @best_fold.setter
    def best_fold(self, new_best_fold):
        if isinstance(new_best_fold, int):
            self._best_fold = new_best_fold
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")
