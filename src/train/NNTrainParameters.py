class NNTrainParameters:

    def __init__(self, batch_size, epochs, device, criterion, optimiser, metrics=() , dict_params_optimiser=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.criterion = criterion
        self.optimiser = optimiser
        self.metrics = metrics # tuple

        self.dict_params_optimiser = dict_params_optimiser

    # SETTERS GETTERS
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        if isinstance(new_batch_size, int) and new_batch_size >= 0:
            self._batch_size = new_batch_size
        else:
            raise TypeError(f"Argument is not an unsigned int.")


    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, new_epochs):
        if isinstance(new_epochs, int) and new_epochs >= 0:
            self._epochs = new_epochs
        else:
            raise TypeError(f"Argument is not an unsigned int.")

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, new_criterion):
        self._criterion = new_criterion

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, new_optimiser):
        self._optimiser = new_optimiser

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics):
        self._metrics = new_metrics
