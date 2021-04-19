from src.training_stopper.Early_stopper import Early_stopper


class Early_stopper_vanilla(Early_stopper):
    # has the parameter has_improved_last_epoch set as true so it will save the new model.
    # never stopped!
    def __init__(self):
        super().__init__(tipee=None, metric_name=None, patience=0, silent=True, delta=0.)

    def __call__(self, *args, **kwargs):
        return False

    def _is_early_stop(self, history, epoch):
        return False
