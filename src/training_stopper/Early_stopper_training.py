from src.training_stopper.Early_stopper import Early_stopper


class Early_stopper_training(Early_stopper):
    def __init__(self, patience=10, silent=True, delta=0.1):
        super().__init__(patience=patience, silent=silent, delta=delta)
        self._lowest_loss = 0

    @staticmethod
    def diff_percent(previous_loss, current_loss):
        return abs(previous_loss - current_loss) / previous_loss

    def _is_early_stop(self, training_losses, epoch):
        """ the criteria is whether the NN is still learning: i.e. the loss moves (in absolute value)."""
        # we compute the relative difference in the training loss of the actual loss wrt to the previous losses.
        # with that, we see if there is any significant improvement in training (significant, more than delta).
        # if there is not, then stop.
        if self._patience < epoch:
            current_loss = training_losses[epoch]
            differences_percent = [self.diff_percent(previous_loss, current_loss)
                                   for previous_loss in training_losses[epoch - self._patience:epoch]]
            cdt = epoch > self._patience and max(differences_percent) < self._delta
            return cdt
