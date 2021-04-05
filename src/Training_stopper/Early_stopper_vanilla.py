from src.Training_stopper.Early_stopper import Early_stopper


class Early_stopper_vanilla(Early_stopper):
    def __init__(self, patience=10, silent=True, delta=0.1, print_func=print):
        super().__init__(patience=patience, silent=silent, delta=delta, print_func=print_func)

    def _is_early_stop(self, training_losses, epoch):
        return False
