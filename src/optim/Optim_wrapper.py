
class Optim_wrapper(object):
    """
    We use wrapper because the initialisation of an optimiser is delayed
    to the moment when the net is created.
    For this reason, the wrapper stores the information of the optimiser.
    """
    def __init__(self, optimiser, optimiser_parameters, scheduler=None, scheduler_parameters=None):
        self.Optimiser = optimiser # class of Optimiser to use. Not initialised.
        self.optimiser_parameters = optimiser_parameters
        self.Scheduler = scheduler # class of Scheduler to use. Not initialised
        self.scheduler_parameters = scheduler_parameters


    def initialise_optimiser(self, later_parameters):
        """ Initialisation of optim."""
        assert self.Optimiser is not None, "No optimiser given."
        assert self.optimiser_parameters is not None, "No parameters for init of optimisers given."

        self.Optimiser = self.Optimiser(later_parameters, **self.optimiser_parameters)

        """ If a scheduler was passed, initialise it. """
        if self._has_scheduler():
            self.Scheduler = self.Scheduler(self.Optimiser, **self.scheduler_parameters)


    def __call__(self, closure):
        self.Optimiser.step(closure)

    def zero_grad(self):
        self.Optimiser.zero_grad()

    def update_learning_rate(self):
        """
            If a scheduler is present, update the learning rate
        """
        if self._has_scheduler():
            self.Scheduler.step()

    def _has_scheduler(self):
        return self.Scheduler is not None and self.scheduler_parameters is not None


