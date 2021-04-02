# parameters

# from useful_functions import *

# for neural networks
import torch.utils.data

import sklearn.model_selection

from src.Neural_Network.Fully_connected_NN import *
from src.Neural_Network.NN_fcts import device, are_at_least_one_None, raise_if_not_all_None, nn_predict
from src.kfold import *
from tqdm import tqdm


def nn_fit(net, X_train_on_device, Y_train_on_device, Y_train,
           params_training,
           training_losses, training_accuracy,
           X_val_on_device=None, Y_val_on_device=None, Y_val=None,
           validation_losses=None, validation_accuracy=None,
           max_through_epoch=None,
           early_stopper_training=None, early_stopper_validation=None,
           compute_accuracy=False,
           silent=False):
    """

    Args:
        net: model
        X_train_on_device:
        Y_train_on_device:
        Y_train:
        params_training:
        training_losses: always passed, numpy array where values are stored.
        training_accuracy: always passed, numpy array where values are stored.
        Passed, even though compute_accuracy is false.
        X_val_on_device:
        Y_val_on_device:
        Y_val:
        validation_losses: always passed, numpy array where values are stored.
        validation_accuracy: always passed, numpy array where values are stored.
        Passed, even though compute_accuracy is false.
        max_through_epoch:
        early_stopper_training:
        early_stopper_validation:
        compute_accuracy: if True, training_accuracy and validation_accuracy are not updated.
        silent: verbose.

    Returns: nothing but updates the value passed, training_losses, training_accuracy,
    validation_losses, validation_accuracy, max_through_epoch.

    """
    # condition if we use validation set.
    list_params_validation = [X_val_on_device, Y_val_on_device,
                              Y_val, validation_losses,
                              validation_accuracy]

    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to are all not None ?
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # create data loader
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_on_device, Y_train_on_device),
                                         batch_size=params_training.batch_size,
                                         shuffle=True,
                                         num_workers=0)  # num_workers can be increased, only under Linux.

    # pick loss function and optimizer
    criterion = params_training.criterion
    optimiser = params_training.optimiser(net.parameters(),
                                          **params_training.dict_params_optimiser)

    for epoch in tqdm(range(params_training.epochs), disable=silent):  # disable unable the print.
        train_loss = 0
        for i, (batch_X, batch_y) in enumerate(loader, 0):
            # closure needed for some algorithm.

            # get batch
            # squeeze batch y in order to have the right format. not the good size for the results
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            def closure():
                # set gradients to zero
                optimiser.zero_grad()

                # Do forward and backward pass
                loss = criterion(net(batch_X), batch_y)
                loss.backward()
                return loss

            # Optimisation step
            optimiser.step(closure=closure)

            # you need to call again criterion unless you do not need the closure.
            train_loss += criterion(net(batch_X), batch_y).item() * batch_X.shape[0]  # weight the loss accordingly

        # Normalize and save the loss over the current epoch:
        training_losses[epoch] = train_loss / (Y_train_on_device.shape[0])
        if compute_accuracy:
            training_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_train_on_device), Y_train)
            # :sklearn can't access data on gpu.

        # Calculate validation loss for the current epoch
        if is_validation_included:
            validation_losses[epoch] = criterion(net(X_val_on_device), Y_val_on_device.squeeze_()).item()
            if compute_accuracy:
                validation_accuracy[epoch] = sklearn.metrics.accuracy_score(nn_predict(net, X_val_on_device), Y_val)
                # :sklearn can't access data on gpu.

            # Calculations to see if it's time to stop early:
            if early_stopper_validation is not None:
                if early_stopper_validation(validation_losses, epoch):
                    break  #: get out of epochs
        if early_stopper_training is not None:
            if early_stopper_training(training_losses, epoch, net):
                break  #: get out of epochs.

    # ~~~~~~~~ end of the for in epoch.
    # we change the value of max_through_epoch:
    max_through_epoch[0] = epoch + 1  #: +1 because it starts at zero so the real value is shifted.


def nn_train(net, data_X, data_Y,
             params_training,
             indic_train_X, indic_train_Y,
             indic_validation_X=None, indic_validation_Y=None,
             early_stopper_training=None, early_stopper_validation=None,
             compute_accuracy=False,
             silent=False):
    """
    Semantics : Given the net, we train it upon data.
    For optimisation reasons, we pass the indices.
    Args:
        net:
        data_X:  tensor
        data_Y:  tensor
        params_training:
        indic_train_X:
        indic_train_Y:
        indic_validation_X:
        indic_validation_Y:
        early_stopper_training: early_stopper_training type,
        early_stopper_validation: early_stopper_validation type,
        compute_accuracy: no impact here, impacts nn_fit.
        silent:

    Returns: Trained net and the data.
    If validation is given:
        returns trained/validation accuracy, trained/validation loss;
    Else:
        returns trained accuracy then loss;
    Whenever the accuracy is not requested, the accuracy vector is zero.
    """
    max_through_epoch = [0]  # : nb of epochs that the NN has back propagated over.
    #: we need to use a container because 0 is immutable, and we want that value to change inside of fit.

    # Prepare Training set
    X_train_on_device = data_X[indic_train_X].to(device)
    Y_train = data_Y[indic_train_Y]  # : useful for using it in order to compute accuracy.
    Y_train_on_device = Y_train.to(device)

    # prepare for iteration over epochs:
    training_losses = np.zeros(params_training.epochs)
    training_accuracy = np.zeros(params_training.epochs)

    # condition if we use validation set:
    list_params_validation = [indic_validation_X, indic_validation_Y]
    is_validation_included = not are_at_least_one_None(list_params_validation)  #: equivalent to are all not None ?
    if not is_validation_included:
        raise_if_not_all_None(list_params_validation)

    # Prepare Validation set if there is any:
    if is_validation_included:
        X_val_on_device = data_X[indic_validation_X].to(device)
        Y_val = data_Y[indic_validation_Y]  # :useful for using it in order to compute accuracy.
        Y_val_on_device = Y_val.to(device)
        validation_losses = np.zeros(params_training.epochs)
        validation_accuracy = np.zeros(params_training.epochs)

        # essentially, we need to check what is the max epoch:
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses, training_accuracy,
               X_val_on_device=X_val_on_device, Y_val_on_device=Y_val_on_device, Y_val=Y_val,
               validation_losses=validation_losses, validation_accuracy=validation_accuracy,
               max_through_epoch=max_through_epoch, early_stopper_training=early_stopper_training,
               early_stopper_validation=early_stopper_validation, compute_accuracy=compute_accuracy, silent=silent)

        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch[0])
        else:
            return (training_accuracy, validation_accuracy,
                    training_losses, validation_losses,
                    max_through_epoch[0])

    # if no validation set
    else:
        nn_fit(net, X_train_on_device, Y_train_on_device, Y_train, params_training, training_losses, training_accuracy,
               max_through_epoch=max_through_epoch, early_stopper_training=early_stopper_training,
               early_stopper_validation=early_stopper_validation, compute_accuracy=compute_accuracy, silent=silent)
        # return loss over epochs and accuracy
        if early_stopper_validation is not None or early_stopper_training is not None:
            return (training_accuracy, training_losses,
                    max_through_epoch[0])
        else:
            return (training_accuracy, training_losses,
                    max_through_epoch[0])


# create main cross validation method
# it returns the score during training,
# but also the best out of the k models, with respect to the accuracy over the whole set.
def nn_kfold_train(data_training_X, data_training_Y,
                   input_size, hidden_sizes, output_size, biases,
                   activation_functions, dropout=0,
                   parameters_for_training=None,
                   early_stopper_validation=None, early_stopper_training=None,
                   nb_split=5, shuffle_kfold=True, percent_validation_for_1_fold=20,
                   compute_accuracy=False,
                   silent=False):
    """

    Args:
        compute_accuracy:
        early_stopper_training:
        data_training_X: tensor
        data_training_Y: tensor
        input_size:
        hidden_sizes:
        output_size:
        biases:
        activation_functions:
        dropout:
        parameters_for_training:
        early_stopper_validation:weight
        nb_split:
        shuffle_kfold:
        percent_validation_for_1_fold:
        silent:

    Returns: net, loss train, loss validation, accuracy train, accuracy validation

    """
    # we distinguish the two cases, but in both we have a list of the result:
    # by inclusivity of else into if compute_accuracy, [0] should be loss and [1] accuracy.
    if compute_accuracy:
        training_data = [np.zeros((nb_split, parameters_for_training.epochs)),
                         np.zeros((nb_split, parameters_for_training.epochs))]
        validation_data = [np.zeros((nb_split, parameters_for_training.epochs)),
                           np.zeros((nb_split, parameters_for_training.epochs))]
    else:
        training_data = [np.zeros((nb_split, parameters_for_training.epochs))]
        validation_data = [np.zeros((nb_split, parameters_for_training.epochs))]

    # The case nb_split = 1: we use the whole dataset for training, without validation:
    if nb_split == 1:
        net = Fully_connected_NN(input_size, hidden_sizes,
                                 output_size, biases,
                                 activation_functions, dropout).to(device)
        # #indices splitting:
        # #WIP CREATE FUNCTION
        training_size = int((100. - percent_validation_for_1_fold) / 100. * data_training_X.shape[0])
        # where validation included.
        if percent_validation_for_1_fold > 0:
            if shuffle_kfold:
                # for the permutation, one could look at https://discuss.pytorch.org/t/shuffling-a-tensor/25422/7:
                # we simplify the expression bc our tensors are in 2D only:
                indices = torch.randperm(                    data_training_X.shape[0])
                #: create a random permutation of the range( nb of data )

                indic_train = indices[:training_size]
                indic_validation = indices[training_size:]
            else:
                indic_train = torch.arange(training_size)
                indic_validation = torch.arange(training_size, data_training_X.shape[1])

            res = nn_train(net,
                           data_X=data_training_X, data_Y=data_training_Y,
                           params_training=parameters_for_training,
                           indic_train_X=indic_train,
                           indic_train_Y=indic_train,
                           indic_validation_X=indic_validation,
                           indic_validation_Y=indic_validation,
                           early_stopper_training=early_stopper_training,
                           early_stopper_validation=early_stopper_validation,
                           compute_accuracy=compute_accuracy,
                           silent=silent)


            training_data[0][0, :] += res[2]
            validation_data[0][0, :] += res[3]
            nb_of_epochs_through = res[4]  # :this number is how many epochs the NN has been trained over.
            # :with such quantity, one does not return a vector full of 0, but only the meaningful data.
            if compute_accuracy:
                training_data[1] += res[0]
                validation_data[1] += res[1]
                return (net,
                        training_data[1][:, :nb_of_epochs_through], validation_data[1][:, :nb_of_epochs_through],
                        training_data[0][:, :nb_of_epochs_through], validation_data[0][:, :nb_of_epochs_through])
            return (net,
                    training_data[0][:, :nb_of_epochs_through], validation_data[0][:, :nb_of_epochs_through])


        # 1-Fold. where validation not included.
        else:
            res = nn_train(net,
                           data_X=data_training_X, data_Y=data_training_Y,
                           params_training=parameters_for_training,
                           indic_train_X=torch.arange(data_training_X.shape[0]),
                           indic_train_Y=torch.arange(data_training_Y.shape[0]),
                           indic_validation_X=None,
                           indic_validation_Y=None,
                           early_stopper_training=early_stopper_training,
                           early_stopper_validation=early_stopper_validation,
                           compute_accuracy=compute_accuracy,
                           silent=silent)

            training_data[0] += res[1]
            nb_of_epochs_through = res[2]  # :this number is how many epochs the NN has been trained over.
            # :with such quantity, one does not return a vector full of 0, but only the meaningful data.
            if compute_accuracy:
                training_data[1] += res[0]
                return (net,
                        training_data[1][:nb_of_epochs_through],
                        training_data[0][:nb_of_epochs_through])
            return (net, training_data[0][:nb_of_epochs_through])
            # : do not return accuracy, only the losses.

    # Kfold for nb_split > 1:
    skfold = sklearn.model_selection.StratifiedKFold(n_splits=nb_split, shuffle=shuffle_kfold, random_state=0)
    # for storing the network:
    performance = 0
    best_net = 0
    nb_max_epochs_through = 0  # :this number is how many epochs the NN has been trained over.
    # :with such quantity, one does not return a vector full of 0, but only the meaningful data.

    for i, (index_training, index_validation) in enumerate(
            skfold.split(data_training_X, data_training_Y)):  # one can use tensors as they are convertible to numpy.
        if not silent:
            print(f"{i + 1}-th Fold out of {nb_split} Folds.")

        net = Fully_connected_NN(input_size, hidden_sizes,
                                 output_size, biases, activation_functions, dropout).to(device)

        # train network and save results
        res = nn_train(net, data_X=data_training_X, data_Y=data_training_Y,
                       params_training=parameters_for_training,
                       indic_train_X=index_training, indic_train_Y=index_training,
                       indic_validation_X=index_validation,
                       indic_validation_Y=index_validation,
                       early_stopper_training=early_stopper_training,
                       early_stopper_validation=early_stopper_validation,
                       compute_accuracy=compute_accuracy,
                       silent=silent)

        if compute_accuracy:
            training_data[1][i, :] += res[0]
            validation_data[1][i, :] += res[1]
        training_data[0][i, :] += res[2]
        validation_data[0][i, :] += res[3]
        nb_of_epochs_through = res[4]

        # storing the best network.
        new_res = res[1][-1]  # the criteria is best validation accuracy at final time.
        if new_res > performance:
            best_net = net
        performance = new_res

        # this number is how many epochs the NN has been trained over.
        # with such quantity, one does not return a vector full of 0, but only the meaningful data.
        if nb_of_epochs_through > nb_max_epochs_through:
            nb_max_epochs_through = nb_of_epochs_through

    return (best_net,
            training_data[1][:, :nb_max_epochs_through], validation_data[1][:, :nb_max_epochs_through],
            training_data[0][:, :nb_max_epochs_through], validation_data[0][:, :nb_max_epochs_through])
