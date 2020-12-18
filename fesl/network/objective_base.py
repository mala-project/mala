from optuna import Trial


class ObjectiveBase:
    """Wraps the training process."""
    def __init__(self, p, dh):
        self.params = p
        self.data_handler = dh

    def __call__(self, trial: Trial):
        """Call needs to be implemented by child classes"""
        raise Exception("No Objective-call implemented.")

    def set_optimal_parameters(self, study):
        """Sets the optimal parameters, needs to be implemented by child classes."""
        raise Exception("No set_optimal_parameters implemented.")
