import optuna
from .objective import Objective


class HyperparameterOptimizer:
    """Serves as an interface to optuna. Enables hyperparameter optimozation."""

    def __init__(self, p):
        self.params = p
        self.study = optuna.create_study(direction=self.params.hyperparameters.direction)

    def perform_study(self, data_handler):
        """Runs the optuna "study" """
        self.study.optimize(Objective(self.params, data_handler), n_trials=self.params.hyperparameters.n_trials)

        print("HPO done with the following values: ")
        self.params.training.learning_rate = self.study.best_params["learning_rate"]
