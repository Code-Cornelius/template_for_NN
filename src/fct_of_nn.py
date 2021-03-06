import torch


def nn_train(Neural_Network,
             data_X, data_Y,
             indices_train_X, indices_train_Y,
             indices_validation_X = None, indices_validation_Y = None,
             parameters_for_training = None,
             early_stop_parameters = None,
             silent = False):
    X = data_X[indices_train_X].values
    y = data_Y[indices_train_Y].values
    validation_X = data_X[indices_validation_X].values
    validation_Y = data_Y[indices_validation_Y].values

    # Prepare Validation set if there is any
    if vX is not None:
        TvX = torch.from_numpy(validation_X).float().to(device)
        Tvy = torch.from_numpy(validation_Y).long().to(device)

        valid_losses = np.zeros(parameters_for_training.epochs)
        valid_acc = np.zeros(parameters_for_training.epochs)

    # Prepare Training set and create data loader
    X_train = torch.from_numpy(X).float()
    Y_train = torch.from_numpy(y).long()
    nn_train = torch.utils.data.TensorDataset(X_train, Y_train)

    loader = torch.utils.data.DataLoader(nn_train, batch_size=batch_size, shuffle=True)
    # pick loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    sgd = torch.optim.SGD(Neural_Network.parameters(), lr=lr)
    # prepare for iteration over epochs
    epoch_losses = np.zeros(parameters_for_training.epochs)
    training_acc = np.zeros(parameters_for_training.epochs)
    min_loss = (-1, 0)

    for epoch in range(parameters_for_training.epochs):
        train_loss = 0
        for i, batch in enumerate(loader, 0):
            # get batch
            bX, by = batch
            # set gradients to zero
            sgd.zero_grad()
            # Do forward and backward pass
            out = Neural_Network(bX)
            by = by.squeeze_()  # not the good size for the results
            loss = criterion(out, by)
            loss.backward()
            # Optimisation step
            sgd.step()

            train_loss += loss.item()
        if epoch % 10 == 1 and not silent:
            print(epoch, "Epochs complete")

        # Normalize and save the loss over the current epoch
        epoch_losses[epoch] = train_loss * batch_size / (y.shape[0])
        training_acc[epoch] = sklearn.metrics.accuracy_score(nn_predict(Neural_Network, X), y)
        # Calculate validation loss for the current epoch
        if vX is not None:
            Tvy = Tvy.squeeze_()
            valid_losses[epoch] = criterion(Neural_Network(TvX), Tvy).item()
            valid_acc[epoch] = sklearn.metrics.accuracy_score(nn_predict(Neural_Network, vX), vy)
            # Calculations to see if it's time to stop early
            if do_earlystop:
                if min_loss[0] is -1 or valid_losses[epoch] < min_loss[1]:
                    min_loss = (epoch, valid_losses[epoch])
                elif epoch - min_loss[0] >= patience:
                    break
    #### end of the for in epoch.
    if not silent:
        print("Training done after %s epochs ! " % epochs)

    # return loss over epochs and accuracy
    if vX is not None:
        if do_earlystop:
            return training_acc, valid_acc, epoch_losses, valid_losses, min_loss[0] + patience
        else:
            return training_acc, valid_acc, epoch_losses, valid_losses
    else:
        return training_acc, epoch_losses


def nn_predict(NeuralNetwork, data_to_predict):
    # do a single predictive forward pass on pnet (takes & returns numpy arrays)
    # Disable dropout:
    NeuralNetwork.train(mode=False)
    # forward pass
    data_predicted = NeuralNetwork(data_to_predict).float().detach().numpy()

    # WIP
    # yhat = pnet(torch.from_numpy(pX.values).float().to(device)).argmax(dim=1).cpu().numpy()
    # WIP

    # Reable dropout:
    NeuralNetwork.train(mode=True)
    return data_predicted


# create main cross validation method
# it returns the score during training, but also the best out of the k models, with respect to the accuracy over the whole set.
def nn_kfold_train(X, y, vX=None, vy=None, k=5, lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                   dropout_p=0,
                   do_earlystop=False, patience=10, shuffle=True, silent=False):
    input_size = len(X.columns)

    mean_training_acc = np.zeros(epochs)
    mean_valid_acc = np.zeros(epochs)
    mean_train_losses = np.zeros(epochs)
    mean_valid_losses = np.zeros(epochs)

    # For early stopping:
    min_epochs = epochs

    # I add case k=1, for my personal testing phase.
    if k == 1:
        net = NeuralNet(input_size, HIDDEN_SIZE, NUM_CLASSES, dropout_p)
        res = nn_train(net, X, y, vX, vy, lr=lr, batch_size=batch_size, epochs=epochs, do_earlystop=do_earlystop,
                       patience=patience, silent=silent)
        mean_training_acc += res[0]
        mean_valid_acc += res[1]
        mean_train_losses += res[2]
        mean_valid_losses += res[3]

        # Average and return results
        mean_training_acc /= k
        mean_valid_acc /= k
        mean_train_losses /= k
        mean_valid_losses /= k
        return net, mean_training_acc[:min_epochs], mean_valid_acc[:min_epochs], mean_train_losses[
                                                                                 :min_epochs], mean_valid_losses[
                                                                                               :min_epochs]

    # Kfold
    cv = sklearn.model_selection.StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=0)
    # for storing the newtork
    performance = 0
    best_net = 0

    for i, (tr, te) in enumerate(cv.split(X, y)):
        if not silent:
            print("Fold %s" % i)

        # Initialize Net with or without dropout
        net = NeuralNet(input_size, HIDDEN_SIZE, NUM_CLASSES, dropout_p).to(device)
        # train network and save results
        res = nn_train(net, X.iloc[tr], y.iloc[tr], vX=X.iloc[te], vy=y.iloc[te], lr=lr, batch_size=batch_size,
                       epochs=epochs, do_earlystop=do_earlystop, patience=patience, silent=silent)
        mean_training_acc += res[0]
        mean_valid_acc += res[1]
        mean_train_losses += res[2]
        mean_valid_losses += res[3]

        # Check if it is time to stop early :
        if do_earlystop:
            if res[4] < min_epochs:
                min_epochs = res[4]

        # storing the network :
        new_res = nn_predict(net, X)
        new_res = sklearn.metrics.accuracy_score(new_res, y)
        if new_res > performance:
            best_net = net
            performance = new_res

    # Average and return results
    mean_training_acc /= k
    mean_valid_acc /= k
    mean_train_losses /= k
    mean_valid_losses /= k
    return best_net, mean_training_acc[:min_epochs], mean_valid_acc[:min_epochs], mean_train_losses[
                                                                                  :min_epochs], mean_valid_losses[
                                                                                                :min_epochs]
