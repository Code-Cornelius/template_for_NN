from src.Training_stopper.Early_stopper import Early_stopper


class Early_stopper_validation(Early_stopper):
    def __init__(self, patience=10, silent=True, delta=0., print_func=print):
        super().__init__(patience=patience, silent=silent, delta=delta, print_func=print_func)

    def is_early_stop(self, validation_losses, epoch):
        if self._lowest_loss * (1 + self._delta) > validation_losses[epoch]:
            return False

        else:
            return True
