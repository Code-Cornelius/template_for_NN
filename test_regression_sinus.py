from src.Neural_Network.NN_fcts import pytorch_device_setting
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


# Number of training samples
n_samples = 5000
# Noise level
sigma = 0.

xx = 2 * np.pi * torch.rand((n_samples,1))
yy = exact_solution(xx) + sigma * torch.randn(xx.shape)

validation_size = int(20. / 100. * n_samples)
training_size = n_samples - validation_size
train_X = xx[:training_size,:]
train_Y = yy[:training_size, :]

validation_X = xx[training_size:, :]
validation_Y = yy[training_size:, :]

input_size = 1
hidden_sizes = [20, 20, 20]
output_size = 1
biases = [True, True, True, True]
activation_functions = [F.relu, F.relu, F.relu]
dropout = 0.4
epochs = 150
batch_size = 1000
optimiser = torch.optim.LBFGS
optimiser = torch.optim.Adam

criterion = nn.MSELoss()

pytorch_device_setting()
dict_optimiser = {"lr": 0.01,
                  "max_iter": 1, "max_eval": 50000,
                  "tolerance_change": 1.0 * np.finfo(float).eps}
dict_optimiser = {"lr": 0.001, "weight_decay" : 0.01}
parameters_for_training = NNTrainParameters(batch_size=batch_size, epochs=epochs,
                                            criterion=criterion, optimiser=optimiser,
                                            dict_params_optimiser=dict_optimiser)

if __name__ == '__main__':
    (net, mean_training_losses) = nn_kfold_train(
        train_X, train_Y, input_size, hidden_sizes, output_size, biases, activation_functions, dropout,
        parameters_for_training=parameters_for_training, early_stopper_validation=None, nb_split=1,
        shuffle_kfold=True, compute_accuracy=False, silent=False)

    nn_plot(mean_training_losses)

    aplot = APlot(how=(1, 1))
    x_test = torch.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_test = exact_solution(x_test).reshape(-1, )

    aplot.uni_plot(nb_ax=0, xx=x_test, yy=y_test, dict_plot_param={"color": "orange",
                                                           "linewidth": 1,
                                                           "label": "Training Data"
                                                           })
    x_pred= torch.linspace(0, 2 * np.pi, 100).reshape(-1,1)
    y_pred = nn_predict(net, x_pred)

    aplot.uni_plot(nb_ax=0, xx=x_pred, yy=y_pred, dict_plot_param={"color": "c",
                                                           "linewidth": 2,
                                                           "label": "Predicted Data"
                                                           })

    # Compute the relative validation error
    relative_error_train = torch.mean((nn_predict(net,train_X) - train_Y) ** 2) / torch.mean(train_Y ** 2)
    print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative validation error
    relative_error_val = torch.mean((nn_predict(net, validation_X) - validation_Y) ** 2) / torch.mean(validation_Y ** 2)
    print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

    # Compute the relative L2 error norm (generalization error)
    relative_error_test = torch.mean((y_pred - y_test) ** 2) / torch.mean(y_test ** 2)
    print("Relative Testing Error: ", relative_error_test.detach().numpy() ** 0.5 * 100, "%")

    aplot.show_legend()
    APlot.show_plot()
