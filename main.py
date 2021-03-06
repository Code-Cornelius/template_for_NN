# parameters
import math
import numpy as np  # maths functions
import matplotlib.pyplot as plt  # for plots
import seaborn as sns  # for the display
import pandas as pd  # for dataframes
import time  # computational time
import scipy.stats as si

# from useful_functions import *

# for neural networks
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import sklearn.model_selection


def euro_vanilla(S, K, T, r, sigma, option='C'):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option == 'C':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K *
                  np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'P':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) -
                  S * si.norm.cdf(-d1, 0.0, 1.0))
    return result


################### generating the data set
np.random.seed(42)
########################################### nb of points
draws = np.power(10, 6)  # number of examples
###########################################
S = np.random.randint(500, 8000, draws) / 100  # S_0. so it is between 5 and 80.

K = np.random.randint(50, 150, draws) * .01 * S  # Strike Price,
# corresponds to a % of the price of the underlying asset.

T = np.random.randint(20, 300, draws) / 100  # 0.2 -> 3 Maturity
r = np.random.randint(1, 2000, draws) / 10000  # from 0.0 to 0.2
sigma = np.random.randint(1, 800, draws) / 1000  # 0 -> 0.8
opt_type = np.random.choice(['C', 'P'], draws)  # choice of call or put.   #['C', 'P']
# generate option prices
opt_price = []
for i in range(draws):
    p = euro_vanilla(S[i], K[i], T[i], r[i], sigma[i], opt_type[i])
    opt_price.append(p)
    if (i % 50000) == 0:
        print('Generated {} Options'.format(i))

# create a dataframe
options = pd.DataFrame({'S': S,
                        'K': K,
                        'T': T,
                        'r': r,
                        'sigma': sigma,
                        'type': opt_type, 'price': opt_price})

options = pd.concat([options, pd.get_dummies(options['type'])], axis=1)
options.drop('type', inplace=True, axis=1)
X, y = options[options.columns.difference(['price'])], options['price']

X_train = X.iloc[draws // 5:]
X_test = X.iloc[:draws // 5]
y_train = y.iloc[draws // 5:]
y_test = y.iloc[:draws // 5]

col_names = X_train.columns

nb_of_points_graph = 101
S = [50]
K = np.linspace(75, 140, nb_of_points_graph) * 0.01 * S
T = [1]
# T = np.random.randint(80, 120, 1) / 100 #
# r = np.random.randint(500, 1000, 1) / 10000 # from 0.05 to 0.1
r = [0.1]
sigma = [0.5]
opt_type = ['C']
# sigma = np.random.randint(1, 50, 1) / 100 # 0 -> 0.5

S = np.full((nb_of_points_graph), S[0])
T = np.full((nb_of_points_graph), T[0])
r = np.full((nb_of_points_graph), r[0])
sigma = np.full((nb_of_points_graph), sigma[0])
opt_type = np.full((nb_of_points_graph - 1), opt_type[0])
# I need both columns in the dataframe so I add a fake final line.
if opt_type[0] == 'C':
    opt_type = np.append(opt_type, 'P')
else:
    opt_type = np.append(opt_type, 'C')

# generate option prices
opt_price = []
for i in range(nb_of_points_graph):
    p = euro_vanilla(S[i], K[i], T[i], r[i], sigma[i], opt_type[i])
    opt_price.append(p)
# create a dataframe
options = pd.DataFrame({'S': S,
                        'K': K,
                        'T': T,
                        'r': r,
                        'sigma': sigma,
                        'type': opt_type, 'price': opt_price})
options = pd.concat([options, pd.get_dummies(options['type'])], axis=1)
options.drop('type', inplace=True, axis=1)
X_graph, y_graph = options[options.columns.difference(['price'])], options['price']
X_graph = X_graph[:-1]
y_graph = y_graph[:-1]



###### scaling process :
import sklearn.preprocessing

scaling = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaling.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=col_names)
X_test_scaled = scaling.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=col_names)
X_graph_scaled = scaling.transform(X_graph)
X_graph_scaled = pd.DataFrame(X_graph_scaled, columns=col_names)



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