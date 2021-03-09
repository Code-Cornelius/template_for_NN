from src.Early_stopper import Early_stopper


class Early_stopper_validation(Early_stopper):
    def __init__(self):
        super().__init__()
        self.highest_loss = None

    def is_early_stop(self, validation_losses, epoch):
        if self.highest_loss is None:
            self.highest_loss = validation_losses[epoch]
            # self.save_checkpoint(val_loss[epoch], neural_network)
            return False
        return self.highest_loss + self.delta < validation_losses[epoch]
