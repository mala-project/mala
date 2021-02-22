import optuna
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase


class HyperOptOptuna(HyperOptBase):
    """Hyperparameter optimizer using Optuna."""

    def __init__(self, params):
        """
        Create a HyperOptOptuna object.

        Parameters
        ----------
        params : fesl.common.parametes.Parameters
            Parameters used to create this hyperparameter optimizer.
        """
        super(HyperOptOptuna, self).__init__(params)
        self.params = params
        self.study = optuna.\
            create_study(direction=self.params.hyperparameters.direction)
        self.objective = None

    def perform_study(self, data_handler):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, optuna is used.

        Parameters
        ----------
        data_handler : fesl.datahandling.data_handler.DataHandler
            datahandler to be used during the hyperparameter optimization.
        """
        self.objective = ObjectiveBase(self.params, data_handler)
        self.study.optimize(self.objective,
                            n_trials=self.params.hyperparameters.n_trials)

        # Return the best lost value we could achieve.
        return self.study.best_value

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        # Parse the parameters from the best trial.
        self.objective.parse_trial_optuna(self.study.best_trial)
