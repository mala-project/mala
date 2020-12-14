import optuna



class HyperparameterOptimizer:
    """Serves as a wrapper to optuna and thus enables hyperparameter optimozation."""

    def __init__(self, p, trainer):
        self.params = p.hyperparameters
        self.study = optuna.create_study(direction=self.params.direction)
        self.trainer = trainer

    def training_wrapper(self, trial):
        """Wraps the training for Optuna."""



    def perform_study(self, data_handler):
