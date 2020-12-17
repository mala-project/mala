from optuna import Trial
from .network import Network
from .trainer import Trainer
from .optuna_parameter import OptunaParameter
from .objective_base import ObjectiveBase


class ObjectiveFeedForward(ObjectiveBase):
    """Wraps the training process."""
    def __init__(self, p, dh):
        super(ObjectiveFeedForward, self).__init__(p, dh)

        # We need to find out if we have to reparametrize the lists with the layers and the activations.
        self.optimize_layer_list = False
        self.optimize_activation_list = False
        par: OptunaParameter
        for par in self.params.hyperparameters.hlist:
            if "ff_neurons_layer" in par.name:
                self.optimize_layer_list = True
            elif "layer_activation" in par.name:
                self.optimize_activation_list = True

    def __call__(self, trial: Trial):
        # parse hyperparameter list.
        print(self.optimize_layer_list, self.optimize_activation_list)
        if self.optimize_layer_list:
            self.params.network.layer_sizes = [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: OptunaParameter
        for par in self.params.hyperparameters.hlist:
            if par.name == "learning_rate":
                self.params.training.learning_rate = par.get_parameter(trial)
            elif "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial))
            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    self.params.network.layer_sizes.append(par.get_parameter(trial))
            else:
                print("Optimization of hyperparameter ", par.name, "not supported at the moment.")
        if self.optimize_layer_list:
            self.params.network.layer_sizes.append(self.data_handler.get_output_dimension())

        # Perform training and report best test loss back to optuna.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss

    def set_optimal_parameters(self, study):
        """Sets the optimal parameters for a feedforward network."""
        # Parse every parameter in the study list.
        if self.optimize_layer_list:
            self.params.network.layer_sizes = [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        for par in study.best_params:
            if par == "learning_rate":
                self.params.training.learning_rate = study.best_params[par]
            if "ff_neurons_layer" in par:
                self.params.network.layer_sizes.append(study.best_params[par])
            elif "layer_activation" in par:
                self.params.network.layer_activations.append(study.best_params[par])

        if self.optimize_layer_list:
            self.params.network.layer_sizes.append(self.data_handler.get_output_dimension())
