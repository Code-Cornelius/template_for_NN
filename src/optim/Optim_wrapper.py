
class Optim_wrapper(object):

    def __init__(self, optim, parameters):
        self.optim = optim
        self.parameters = parameters

    def initialise_optim(self, later_parameters):
        assert self.optim is not None
        assert self.parameters is not None

        self.optim = self.optim(later_parameters, **self.parameters)

    def __call__(self, closure):
        self.optim.step(closure)

    def zero_grad(self):
        self.optim.zero_grad()