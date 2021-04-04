# todo requirement
#  make the workers work, in particular check if they work in Linux.
#  and save the model and use it to check the accuracy total.

# for neural networks
from src.Neural_Network.NN_fcts import pytorch_device_setting
from src.Neural_Network.NN_kfold_training import nn_kfold_train
from src.Neural_Network.NN_plots import *
from src.Neural_Network.NN_training import *
from src.Neural_Network.NNTrainParameters import *
from src.Neural_Network.Fully_connected_NN import *

# set seed for pytorch.
from src.Training_stopper.Early_stopper_training import Early_stopper_training
from src.Training_stopper.Early_stopper_validation import Early_stopper_validation

torch.manual_seed(42)
np.random.seed(42)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 10000
# Noise level
sigma = 0.01
pytorch_device_setting()
SILENT = False
early_stop_train = Early_stopper_training(patience=20, silent=SILENT, delta=0.1)
early_stop_valid = Early_stopper_validation(patience=20, silent=SILENT, delta=0.1)
#############################


from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = pd.DataFrame(train_X.reshape(60000, 28 * 28))
train_Y = pd.DataFrame(train_y)

test_X = pd.DataFrame(test_X.reshape(10000, 28 * 28))
test_Y = pd.DataFrame(test_y)

train_X = train_X[:n_samples]
train_Y = train_Y[:n_samples]
test_X = test_X[:n_samples]
test_Y = test_Y[:n_samples]

train_X = torch.from_numpy(train_X.values).float()
train_Y = torch.from_numpy(train_Y.values).long().squeeze()
test_X = torch.from_numpy(test_X.values).float()
test_Y = torch.from_numpy(test_Y.values).long().squeeze()

if __name__ == '__main__':
    # config of the architecture:
    input_size = 28 * 28
    hidden_sizes = [200, 200]
    output_size = 10
    biases = [True, True, True]
    activation_functions = [F.relu, F.relu]
    dropout = 0.4
    epochs = 200
    batch_size = 2000
    optimiser = torch.optim.SGD
    criterion = nn.CrossEntropyLoss()
    dict_optimiser = {"lr": 0.001, "weight_decay": 0.01}

    parameters_for_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                                criterion=criterion, optimiser=optimiser,
                                                dict_params_optimiser=dict_optimiser)
    parametrized_NN = factory_parametrised_FC_NN(input_size=input_size, list_hidden_sizes=hidden_sizes,
                                                 output_size=output_size,
                                                 list_biases=biases, activation_functions=activation_functions,
                                                 dropout=dropout, predict_fct=lambda out: torch.max(out, 1)[1])

    # training
    print(" ~~~~~~~~~~Example 1 : Split 1~~~~~~~~~~ ")
    (net, mean_training_accuracy, mean_validation_accuracy,
     mean_training_losses, mean_validation_losses) = nn_kfold_train(train_X, train_Y,
                                                                    parametrized_NN,
                                                                    parameters_training=parameters_for_training,
                                                                    early_stopper_validation=None, nb_split=1,
                                                                    shuffle_kfold=True,
                                                                    percent_validation_for_1_fold=20,
                                                                    compute_accuracy=True,
                                                                    silent=False)

    nn_plot_train_loss_acc(mean_training_losses, mean_validation_losses, mean_training_accuracy,
                           mean_validation_accuracy)

    print(" ~~~~~~~~~~Example 2 : Split 1 with both stopper~~~~~~~~~~ ")
    (net, mean_training_accuracy, mean_validation_accuracy,
     mean_training_losses, mean_validation_losses) = nn_kfold_train(train_X, train_Y,
                                                                    parametrized_NN,
                                                                    parameters_training=parameters_for_training,
                                                                    early_stopper_training=early_stop_train,
                                                                    early_stopper_validation=early_stop_valid,
                                                                    shuffle_kfold=True, nb_split=1,
                                                                    percent_validation_for_1_fold=20,
                                                                    compute_accuracy=True,
                                                                    silent=False)

    nn_plot_train_loss_acc(mean_training_losses, mean_validation_losses, mean_training_accuracy,
                           mean_validation_accuracy)

    print(" ~~~~~~~~~~Example 3 : Split 5~~~~~~~~~~ ")
    (net, mean_training_accuracy, mean_validation_accuracy,
     mean_training_losses, mean_validation_losses) = nn_kfold_train(train_X, train_Y,
                                                                    parametrized_NN,
                                                                    parameters_training=parameters_for_training,
                                                                    early_stopper_validation=None, nb_split=5,
                                                                    shuffle_kfold=True,
                                                                    compute_accuracy=True,
                                                                    silent=False)

    nn_plot_train_loss_acc(mean_training_losses, mean_validation_losses, mean_training_accuracy,
                           mean_validation_accuracy)

    print(" ~~~~~~~~~~Example 4 : no validation for 1 split ~~~~~~~~~~ ")
    (net, mean_training_accuracy,
     mean_training_losses) = nn_kfold_train(train_X, train_Y,
                                            parametrized_NN,
                                            parameters_training=parameters_for_training,
                                            early_stopper_validation=None, nb_split=1,
                                            shuffle_kfold=True,
                                            percent_validation_for_1_fold=0,
                                            compute_accuracy=True,
                                            silent=False)

    nn_plot_train_loss_acc(mean_training_losses, None, mean_training_accuracy)

    confusion_matrix_creator(train_Y, nn_predict(net, train_X), range(10), title="Training Set")
    confusion_matrix_creator(test_Y, nn_predict(net, test_X), range(10), title="Test Set")
    APlot.show_plot()
