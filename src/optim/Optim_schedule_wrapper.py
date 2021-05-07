from optim.Optim_wrapper import Optim_wrapper


class Optim_scheduler_wrapper(Optim_wrapper):

    def __init__(self, optim, parameters):
        super().__init__(optim, parameters)

    def initialise_optim(self, later_parameters):
        if self.optim is not None and self.parameters is not None:
            super().initialise_optim(later_parameters)

    def __call__(self, **kwargs):
        if self.optim is not None and self.parameters is not None:
            self.optim.step()