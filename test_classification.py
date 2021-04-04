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
torch.manual_seed(42)
np.random.seed(42)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 10000
# Noise level
sigma = 0.01
pytorch_device_setting()
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
    epochs = 20
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

    nn_plot_train_loss_acc(mean_training_losses, mean_validation_losses, mean_training_accuracy, mean_validation_accuracy)

    print(" ~~~~~~~~~~Example 2 : Split 5~~~~~~~~~~ ")
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

    print(" ~~~~~~~~~~Example 3 : no validation for 1 split ~~~~~~~~~~ ")
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

# TODO WHAT IS THIS
# analyze_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5, epochs=30, silent=True)
#
# # Activation Function
# batch_size = 128
# learning_rate = 0.005
# epochs = 30
# hidden_size = 16
# num_layers = 2
# dropout = 0
# norm = False
# activ_function = "tanh"
# version = 0
# optim = "sgd"
#
# analyze_convolution_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5,
#                                    batch_size, learning_rate, epochs,
#                                    hidden_size, num_layers, dropout, norm, activ_function, version, optim, True)
