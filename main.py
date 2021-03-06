analyze_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5, epochs=30, silent=True)

# Activation Function
batch_size = 128
learning_rate = 0.005
epochs = 30
hidden_size = 16
num_layers = 2
dropout = 0
norm = False
activ_function = "tanh"
version = 0
optim = "sgd"

analyze_convolution_neural_network(data_train_X, data_train_Y, data_test_X, data_test_Y, 5,
                                   batch_size, learning_rate, epochs,
                                   hidden_size, num_layers, dropout, norm, activ_function, version, optim, True)