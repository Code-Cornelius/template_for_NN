from src.training_stopper.Early_stopper import Early_stopper


class Early_stopper_validation(Early_stopper):
    def __init__(self, patience=10, silent=True, delta=0.):
        super().__init__(patience=patience, silent=silent, delta=delta)

    def _is_early_stop(self, validation_losses, epoch):
        """ the critirea is whether the NN is not overfitting: i.e. the validation loss is decreasing. If delta is too big, then a model where the validation is constant keeps training !"""
        if self._lowest_loss * (1 + self._delta) > validation_losses[epoch]:
            return False
        else:
            return True