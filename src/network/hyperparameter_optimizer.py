import optuna
from .objective import Objective


class HyperparameterOptimizer:
    """Serves as an interface to optuna. Enables hyperparameter optimozation."""

    def __init__(self, p):
        self.params = p
        self.study = optuna.create_study(direction=self.params.hyperparameters.direction)

        # Needed to reparametrize the network architecture.
        self.inputdim = 0
        self.outputdim = 0

    def perform_study(self, data_handler):
        """Runs the optuna "study" """

        # Needed to reparametrize the network architecture.
        self.inputdim = data_handler.get_input_dimension()
        self.outputdim = data_handler.get_output_dimension()

        # Perform the study.
        self.study.optimize(Objective(self.params, data_handler), n_trials=self.params.hyperparameters.n_trials)

        # Print the results.
        print("Hyperparameter optimization done with the following values: ")
        for hyppar in self.study.best_params:
            print(hyppar, ": ", self.study.best_params[hyppar])

    def set_optimal_parameters(self):
        """Writes the best set of parameters found by an optuna study back into
        the parameters object."""

        # Parse every parameter in the study list.
        for par in self.study.best_params:
            if par == "learning_rate":
                self.params.training.learning_rate = self.study.best_params[par]
            elif par == "number_of_hidden_neurons":
                self.params.network.layer_sizes = [self.inputdim, self.study.best_params[par],
                                                       self.outputdim]
            elif par == "layer_activations":
                self.params.network.layer_activations = [self.study.best_params[par]]


