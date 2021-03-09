from src.Early_stopper import Early_stopper


class Early_stopper_training(Early_stopper):

    def __init__(self):
        super().__init__()

    def is_early_stop(self, training_losses, epoch):
        # if the percentage of change of the actual loss wrt to any loss
        # for the 20 previous loss is less than 10%, then stop.
        cdt = epoch > 20 and all(
            difference_percentage < 0.1 for difference_percentage in
            [abs(previous_loss - training_losses[epoch]) / previous_loss for previous_loss in
             training_losses[epoch - 20:epoch]
             ]
        )
        return cdt
