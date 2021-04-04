from src.Neural_Network.NN_fcts import pytorch_device_setting
from src.Neural_Network.NN_kfold_training import nn_kfold_train
from src.Neural_Network.NN_plots import *
from src.Neural_Network.NN_training import *
from src.Neural_Network.NNTrainParameters import *
from src.Neural_Network.Fully_connected_NN import *

# set seed for pytorch.
torch.manual_seed(42)
np.random.seed(42)


# Define the exact solution
def exact_solution(x):
    return torch.sin(x)

############################## GLOBAL PARAMETERS
# Number of training samples
n_samples = 1000
# Noise level
sigma = 0.01
pytorch_device_setting()
#############################
plot_xx = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
plot_yy = exact_solution(plot_xx).reshape(-1, )
plot_yy_noisy = (exact_solution(plot_xx) + sigma * torch.randn(plot_xx.shape) ).reshape(-1, )

xx = 2 * np.pi * torch.rand((n_samples,1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

training_size = int(90. / 100. * n_samples)
train_X = xx[:training_size,:]
train_Y = yy[:training_size, :]

testing_X = xx[training_size:, :]
testing_Y = yy[training_size:, :]



if __name__ == '__main__':
    # config of the architecture:
    input_size = 1
    hidden_sizes = [20, 20, 20]
    output_size = 1
    biases = [True, True, True, True]
    activation_functions = [torch.tanh, torch.tanh, torch.relu]
    dropout = 0.
    epochs = 100
    batch_size = 200
    optimiser = torch.optim.Adam
    criterion = nn.MSELoss()

    dict_optimiser = {"lr": 0.001, "weight_decay": 0.0000001}
    parameters_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                            criterion=criterion, optimiser=optimiser,
                                            dict_params_optimiser=dict_optimiser)
    parametrized_NN = factory_parametrised_FC_NN(input_size=input_size, list_hidden_sizes=hidden_sizes,
                                                 output_size=output_size,
                                                 list_biases=biases, activation_functions=activation_functions,
                                                 dropout=dropout, predict_fct = None)

    # training
    print(" ~~~~~~~~~~Example 1 : Split 1~~~~~~~~~~ ")
    (net, mean_training_losses, mean_validation_losses) = nn_kfold_train(train_X, train_Y,
                                                                         parametrized_NN,
                                                                         parameters_training=parameters_training,
                                                                         early_stopper_validation=None, nb_split=1,
                                                                         shuffle_kfold=True,
                                                                         percent_validation_for_1_fold=20,
                                                                         compute_accuracy=False,
                                                                         silent=False)
    nn_plot_train_loss_acc(training_loss = mean_training_losses, validation_loss=mean_validation_losses)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)

    # todo IMPLEMENT NOT STRATIFIED K-FOLD
    # print(" ~~~~~~~~~~Example 2 : Split 5~~~~~~~~~~ ")
    # (net, mean_training_losses, mean_validation_losses) = nn_kfold_train(train_X, train_Y,
    #                                                                      parametrized_NN,
    #                                                                      parameters_training=parameters_training,
    #                                                                      early_stopper_validation=None, nb_split=5,
    #                                                                      shuffle_kfold=True,
    #                                                                      compute_accuracy=False, silent=False)
    # nn_plot_train_loss_acc(training_loss=mean_training_losses, validation_loss=mean_validation_losses)
    # nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    # nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)

    print(" ~~~~~~~~~~Example 3 : no validation for 1 split ~~~~~~~~~~ ")
    (net, mean_training_losses) = nn_kfold_train(train_X, train_Y, parametrized_NN,
                                                                         parameters_training=parameters_training,
                                                                         early_stopper_validation=None, nb_split=1,
                                                                         shuffle_kfold=True,
                                                                         percent_validation_for_1_fold=0,
                                                                         compute_accuracy=False, silent=False)
    nn_plot_train_loss_acc(training_loss=mean_training_losses)
    nn_plot_prediction_vs_true(net, plot_xx, plot_yy, plot_yy_noisy)
    nn_print_errors(net, train_X, train_Y, testing_X, testing_Y)




    APlot.show_plot()

