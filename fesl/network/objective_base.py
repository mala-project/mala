from abc import abstractmethod

from optuna import Trial

from fesl.common.parameters import Parameters
from fesl.network.optuna_parameter import OptunaParameter


def copy_training_parameters(parameters: Parameters, trial: Trial) -> Parameters:
    """Mutates passed parameters ¯\_(ツ)_/¯"""
    par: OptunaParameter
    for par in parameters.hyperparameters.hlist:
        if par.name == "learning_rate":
            parameters.training.learning_rate = par.get_parameter(trial)
        elif "layer_activation" in par.name:
            parameters.network.layer_activations.append(par.get_parameter(trial))
        elif "ff_neurons_layer" in par.name:
            if parameters.network.nn_type == "feed-forward":
                parameters.network.layer_sizes.append(par.get_parameter(trial))
        else:
            raise Exception(f"Optimization of hyperparameter {par.name} not supported at the moment.")
    return parameters


class ObjectiveBase:
    """Wraps the training process."""
    def __init__(self, p, dh):
        self.params = p
        self.data_handler = dh

        # We need to find out if we have to reparametrize the lists with the layers and the activations.
        self.optimize_layer_list = any(map(
            lambda p: "ff_neurons_layer" in p.name,
            self.params.hyperparameters.hlist
        ))
        self.optimize_activation_list = any(map(
            lambda p: "layer_activation" in p.name,
            self.params.hyperparameters.hlist
        ))

    @abstractmethod
    def __call__(self, trial: Trial):
        """Call needs to be implemented by child classes"""

        # parse hyperparameter list.
        if self.optimize_layer_list:
            self.params.network.layer_sizes = [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        copy_training_parameters(self.params, trial)

        if self.optimize_layer_list:
            self.params.network.layer_sizes.append(self.data_handler.get_output_dimension())

    def set_optimal_parameters(self, study):
        """Sets the optimal parameters, needs to be implemented by child classes."""
        raise Exception("No set_optimal_parameters implemented.")
