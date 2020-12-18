import optuna
from .objective_interface import ObjectiveInterface


class HyperparameterOptimizer:
    """Serves as an interface to optuna. Enables hyperparameter optimozation."""

    def __init__(self, p):
        self.params = p
        self.study = optuna.create_study(direction=self.params.hyperparameters.direction)

        # Needed to reparametrize the network architecture.
        self.inputdim = 0
        self.outputdim = 0
        self.objective = None

    def perform_study(self, data_handler):
        """Runs the optuna "study" """

        # Needed to reparametrize the network architecture.
        self.inputdim = data_handler.get_input_dimension()
        self.outputdim = data_handler.get_output_dimension()

        # Perform the study.
        self.objective = ObjectiveInterface(self.params, data_handler)
        self.study.optimize(self.objective, n_trials=self.params.hyperparameters.n_trials)

        # Print the results.
        print("Hyperparameter optimization done with the following values: ")
        for hyppar in self.study.best_params:
            print(hyppar, ": ", self.study.best_params[hyppar])

    def set_optimal_parameters(self):
        """Writes the best set of parameters found by an optuna study back into
        the parameters object."""

        # Option for additional code.

        # Call the function from objective.
        self.objective.set_optimal_parameters(self.study)

