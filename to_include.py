# todo depends on activation function
# section ######################################################################
#  #############################################################################
#

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)

def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


# section ######################################################################
#  #############################################################################
#
# Compute the relative L2 error norm (generalization error)
relative_error_test = torch.mean((y_test_pred - y_test) ** 2) / torch.mean(y_test ** 2)
print("Relative Error Test: ", relative_error_test.detach().numpy() * 100, "%")
# section ######################################################################
#  #############################################################################
#
if opt_type == "ADAM":
    optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
elif opt_type == "LBFGS":
    optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000,
                             tolerance_change=1.0 * np.finfo(float).eps)
else:
    raise ValueError("Optimizer not recognized")

history = fit(my_network, training_set, x_val, y_val, n_epochs, optimizer_, p=2, verbose=False)

x_test = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
y_test = exact_solution(x_test).reshape(-1, )
y_val = y_val.reshape(-1, )
y_train = y_train.reshape(-1, )

y_test_pred = my_network(x_test).reshape(-1, )
y_val_pred = my_network(x_val).reshape(-1, )
y_train_pred = my_network(x_train).reshape(-1, )

# Compute the relative validation error
relative_error_train = torch.mean((y_train_pred - y_train) ** 2) / torch.mean(y_train ** 2)
print("Relative Training Error: ", relative_error_train.detach().numpy() ** 0.5 * 100, "%")

# Compute the relative validation error
relative_error_val = torch.mean((y_val_pred - y_val) ** 2) / torch.mean(y_val ** 2)
print("Relative Validation Error: ", relative_error_val.detach().numpy() ** 0.5 * 100, "%")

# Compute the relative L2 error norm (generalization error)
relative_error_test = torch.mean((y_test_pred - y_test) ** 2) / torch.mean(y_test ** 2)
print("Relative Testing Error: ", relative_error_test.detach().numpy() ** 0.5 * 100, "%")

return relative_error_train.item(), relative_error_val.item(), relative_error_test.item()

# section ######################################################################
#  #############################################################################
#
#todo create a function that given the dictionnary, create a list of NNParameters
settings = list(itertools.product(*network_properties.values()))
for set_num, setup in enumerate(settings):
    print("###################################", set_num, "###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "regularization_exp": setup[2],
        "regularization_param": setup[3],
        "batch_size": setup[4],
        "epochs": setup[5],
        "optimizer": setup[6],
        "init_weight_seed": setup[7]
    }
