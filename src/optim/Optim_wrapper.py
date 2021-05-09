
class Optim_wrapper(object):
    """
    We use wrapper because the initialisation of an optimiser is delayed
    to the moment when the net is created.
    For this reason, the wrapper stores the information of the optimiser.
    """
    def __init__(self, optim, parameters):
        self.Optim = optim # class of Optimiser to use. Not initialised.
        self.parameters = parameters

    def initialise_optim(self, later_parameters):
        """ Initialisation of optim."""
        assert self.Optim is not None, "No optimiser given."
        assert self.parameters is not None, "No parameters for init of optimisers given."

        self.Optim = self.Optim(later_parameters, **self.parameters)

    def __call__(self, closure):
        self.Optim.step(closure)

    def zero_grad(self):
        self.Optim.zero_grad()