import pandas as pd
from priv_lib_estimator import Estimator
from priv_lib_util.tools.src.function_json import unzip_json, zip_json
from priv_lib_util.tools.src.function_writer import list_of_dicts_to_json


import json

class Estim_history(Estimator):
    BASE_COLUMN_NAMES = ['fold', 'epoch']

    def __init__(self, df=None, metric_names=None, validation=True):
        if df is not None:
            # initialise with dataframe from csv
            super().__init__(df)

        else:
            df_column_names = Estim_history._generate_all_column_names(metric_names, validation)
            data_frame = pd.DataFrame(columns=df_column_names)
            super().__init__(data_frame)

            self.validation = validation
            self.best_epoch = []

            # TODO: collect training parameters
            self.training_parameters = {}

    @staticmethod
    def _generate_column_name(base_name, validation=False):
        """
        Generate the column name based on a metric name and its use case (training or validation)
        Args:
            base_name: the name of the metric
            validation: boolean representing whether the metric is a result of validation or training

        Returns:
            The formatted column name
        """
        return f"{base_name}_" + ('validation' if validation else 'training')

    @staticmethod
    def _generate_all_column_names(metric_names, validation):
        """
            Generate all the column names for the dataframe
        Args:
            metric_names: A list of the names of the metrics used
            validation: A boolean representing whether validation is used during training

        Returns:
            A list of all the column names for the dataframe (Including the base columns)
        """
        df_column_names = Estim_history.BASE_COLUMN_NAMES.copy()
        df_column_names.append(Estim_history._generate_column_name("loss"))

        for metric_name in metric_names:
            df_column_names.append(Estim_history._generate_column_name(metric_name))

        if validation:
            df_column_names.append(Estim_history._generate_column_name("loss", validation=True))

            for metric_name in metric_names:
                df_column_names.append(Estim_history._generate_column_name(metric_name, validation=True))

        return df_column_names


    def append_history(self, history, best_epoch, fold_number):
        """
            Append information from history to the estimator
        Args:
            history: history of the training
            fold_number: the fold number the history corresponds to

        Returns:
            Void
        """
        self.best_epoch.append(best_epoch)
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

        # add the epoch number to the translated history
        number_of_epochs = len(history['training']['loss'])
        translated_history['epoch'] = [*range(number_of_epochs)]

        # collect training information
        for key, value in history['training'].items():
            new_key = self._generate_column_name(key)
            translated_history[new_key] = value.tolist()

        assert ('validation' in history) == self.validation, "The information about validation in estimator " \
                                                           "is not reflected in history"
        # collect validation information if present
        if 'validation' in history:
            for key, value in history['validation'].items():
                new_key = self._generate_column_name(key, validation=True)
                translated_history[new_key] = value.tolist()


        # add the fold number to the history
        translated_history['fold'] = [fold_number] * number_of_epochs

        return translated_history

    def get_value_at_index(self, fold, epoch, column):
        index = self._index_mask(fold, epoch)
        return self._df[column][index].values[0]

    def get_values_at_fold(self, fold, column):
        index = self._fold_mask(fold)
        return self._df[column][index].values

    def get_values_at_column(self, column):
        return self._df[column].values

    def _index_mask(self, fold, epoch):
        return (self._df['fold'] == fold) & (self._df['epoch'] == epoch)

    def _fold_mask(self, fold):
        return self._df['fold'] == fold

    def number_of_folds(self):
        return self._df['fold'].max() + 1

    def number_of_epochs(self):
        return self._df['epoch'].max() + 1

    def to_json(self, path):
        self._df.validation = self.validation

        json_df = self._df.to_json(orient='split')
        parsed = json.loads(json_df)
        parsed['attrs'] = {
            'validation': self.validation,
            'best_epoch': self.best_epoch,
        }

        zipped = zip_json(parsed)

        with open(path, 'w') as file:
            json.dump(zipped, file)

    @classmethod
    def from_file(cls, path):

        with open(path, 'r') as file:
            df_info = json.load(file)
            unzipped = unzip_json(df_info)
            attrs = unzipped['attrs']
            del unzipped['attrs']

        with open(path, 'w') as file:
            json.dump(unzipped, file)

        dataframe = pd.read_json(path, orient='split')
        estimator = cls(dataframe)

        estimator.validation = attrs['validation']
        estimator.best_epoch = attrs['best_epoch']
        return estimator

