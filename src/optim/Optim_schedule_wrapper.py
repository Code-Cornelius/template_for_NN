from optim.Optim_wrapper import Optim_wrapper


class Optim_scheduler_wrapper(Optim_wrapper):
    """
    Subclass for learning rate schedulers.
    """
    def __init__(self, optim, parameters):
        super().__init__(optim, parameters)

    def initialise_optim(self, later_parameters):
        """ initialise the optimiser iff the parameters are not None. """
        if self.Optim is not None and self.parameters is not None:
            super().initialise_optim(later_parameters)

    def __call__(self, **kwargs):
        if self.Optim is not None and self.parameters is not None:
            self.Optim.step()