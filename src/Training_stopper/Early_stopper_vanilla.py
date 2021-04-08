from src.Training_stopper.Early_stopper import Early_stopper


class Early_stopper_vanilla(Early_stopper):
    # has the parameter has_improved_last_epoch set as true so it will save the new model.
    # never stopped!
    def __init__(self):
        super().__init__(patience=0, silent=True, delta=0.)

    def _is_early_stop(self, training_losses, epoch):
        return False
