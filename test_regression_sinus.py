# Number of training samples
n_samples = 100
# Noise level
sigma = 0.0

x = 2 * np.pi * torch.rand((n_samples, 1))
y = exact_solution(x) + sigma * torch.randn(x.shape)

network_properties = {
    "hidden_layers": [2, 4],
    "neurons": [5, 20],
    "regularization_exp": [2],
    "regularization_param": [0, 1e-4],
    "batch_size": [n_samples],
    "epochs": [1000],
    "optimizer": ["LBFGS"],
    "init_weight_seed": [567, 34, 134]
}

settings = list(itertools.product(*network_properties.values()))

i = 0

train_err_conf = list()
val_err_conf = list()
test_err_conf = list()
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

    relative_error_train_, relative_error_val_, relative_error_test_ = run_configuration(setup_properties, x, y)
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)
    test_err_conf.append(relative_error_test_)

print(train_err_conf, val_err_conf, test_err_conf)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)
test_err_conf = np.array(test_err_conf)

plt.figure(figsize=(16, 8))
plt.grid(True, which="both", ls=":")
plt.scatter(np.log10(train_err_conf), np.log10(test_err_conf), marker="*", label="Training Error")
plt.scatter(np.log10(val_err_conf), np.log10(test_err_conf), label="Validation Error")
plt.xlabel("Selection Criterion")
plt.ylabel("Generalization Error")
plt.title(r'Validation - Training Error VS Generalization error ($\sigma=0.0$)')
plt.legend()
plt.savefig("sigma.png", dpi=400)
plt.show()
