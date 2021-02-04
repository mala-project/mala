import optuna
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase


class HyperOptOptuna(HyperOptBase):
    """Perform hyperparameter optimization using optuna."""

    def __init__(self, p):
        super(HyperOptOptuna,self).__init__(p)
        self.params = p
        self.study = optuna.create_study(direction=self.params.hyperparameters.direction)
        self.objective = None

    def perform_study(self, data_handler):
        """Runs the optuna "study" """

        # Perform the study.
        self.objective = ObjectiveBase(self.params, data_handler)
        self.study.optimize(self.objective, n_trials=self.params.hyperparameters.n_trials)

        # Return the best lost value we could achieve.
        return self.study.best_value

    def set_optimal_parameters(self):
        """Writes the best set of parameters found by an optuna study back into
        the parameters object."""

        # Parse the parameters from the best trial.
        self.objective.parse_trial_optuna(self.study.best_trial)

