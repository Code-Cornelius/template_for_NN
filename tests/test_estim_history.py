from unittest import TestCase

from nn_classes.estimator.history.estim_history import Estim_history
import numpy as np
import os

from nn_classes.estimator.hyper_parameters.estim_hyper_param import Estim_hyper_param
from nn_train.kfold_training import _translate_history_to_dataframe

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_history = {
    'training': {
        'loss': np.array(range(10)),
        'L1': np.array(range(10, 20)),
        'L2': np.array(range(20, 30))
    },
    'validation': {
        'loss': np.array(range(30, 40)),
        'L1': np.array(range(40, 50)),
        'L2': np.array(range(50, 60))
    },
    'best_epoch': 2
}
metric_names = ['L1', 'L2']

flattened_history = {
    'epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'loss_training': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'L1_training': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'L2_training': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'loss_validation': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    'L1_validation': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'L2_validation': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    'fold': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

class Test_estim_history(TestCase):

    def test_create_estim_with_validation_has_correct_column_names(self):
        estimator = Estim_history(metric_names=metric_names, validation=True)

        column_names = ['fold', 'epoch', 'loss_training', 'L2_training', 'L3_training',
                        'loss_validation', 'L2_validation', 'L3_validation'].sort()

        df_column_names = estimator._df.columns.values.sort()

        assert column_names == df_column_names

    def test_create_estim_without_validation_has_correct_column_names(self):
        estimator = Estim_history(metric_names=metric_names, validation=False)

        column_names = ['fold', 'epoch', 'loss_training', 'L2_training', 'L3_training'].sort()

        df_column_names = estimator._df.columns.values.sort()

        assert column_names == df_column_names

    def test_translate_flattens_the_history(self):
        estimator = Estim_history(metric_names=metric_names, validation=True)

        translated_history = estimator._translate_history_to_dataframe(history, 1)

        assert translated_history == flattened_history

    def test_append_history_from_folds_to_estim(self):
        estimator = Estim_history(metric_names=metric_names, validation=True)

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, 2)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, 3)

        df = estimator._df

        assert df.shape == (20, 8)


    def test_to_csv(self):
        file_name = "test.json"

        path = os.path.join(ROOT_DIR, file_name)
        estimator = Estim_history(metric_names=metric_names, validation=True)

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, 2)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, 3)

        estimator.to_json(path)

        new_estim = Estim_history.from_json(path)

        print(new_estim)

    def test_test_Test(self):
        folder_name = "param_train_test"
        path = os.path.join(ROOT_DIR, folder_name)

        estimator = Estim_history(metric_names=metric_names, validation=True,
                                  hyper_params={'optim': 'adam',
                                                       'lr': 0.7,
                                                       'layers': [12, 10]})

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, 2)
        estimator.best_fold = 0

        file_path1 = os.path.join(path, "estim_1")
        estimator.to_json(file_path1)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, 3)

        estimator.best_fold = 1
        estimator.slice_best_fold()

        file_path2 = os.path.join(path, "estim_2")
        estimator.to_json(file_path2)

        estimator_param = Estim_hyper_param.from_folder(path, "loss_validation")

        print(estimator)
        print(estimator_param)