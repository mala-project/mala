from optuna import Trial
from .hyperparameter_optuna import HyperparameterOptuna
from .hyperparameter_oat import HyperparameterOAT
from .network import Network
from .trainer import Trainer

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

        self.trial_type = self.params.hyperparameters.hyper_opt_method

    def __call__(self, trial):
        """Call needs to be implemented by child classes"""

        # Parse the parameters included in the trial.
        self.parse_trial(trial)

        # Perform training and report best test loss.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss

    def parse_trial(self, trial):
        if self.trial_type == "optuna":
            self.parse_trial_optuna(trial)
        elif self.trial_type == "oat":
            self.parse_trial_oat(trial)
        else:
            raise Exception("Cannot parse trial, unknown hyperparameter optimization method.")

    def parse_trial_optuna(self, trial: Trial):
        """
        Parse an optuna style trial into the params attribute.
        """


        if self.optimize_layer_list:
            self.params.network.layer_sizes = [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: HyperparameterOptuna
        for par in self.params.hyperparameters.hlist:
            if par.name == "learning_rate":
                self.params.training.learning_rate = par.get_parameter(trial)
            elif "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial))
            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    self.params.network.layer_sizes.append(par.get_parameter(trial))
            elif "trainingtype" in par.name:
                self.params.training.trainingtype = par.get_parameter(trial)
            else:
                raise Exception("Optimization of hyperparameter ", par.name, "not supported at the moment.")
        if self.optimize_layer_list:
            self.params.network.layer_sizes.append(self.data_handler.get_output_dimension())


    def parse_trial_oat(self, trial):
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: HyperparameterOAT
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial,factor_idx))
            elif "trainingtype" in par.name:
                self.params.training.trainingtype= par.get_parameter(trial, factor_idx)
            else:
                raise Exception("Optimization of hyperparameter ", par.name, "not supported at the moment.")