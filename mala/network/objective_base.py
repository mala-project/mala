"""Objective function for all training based hyperparameter optimizations."""
from optuna import Trial
from .hyperparameter_optuna import HyperparameterOptuna
from .hyperparameter_oat import HyperparameterOAT
from .network import Network
from .trainer import Trainer


class ObjectiveBase:
    """
    Represents the objective function of a training process.

    This is usually the result of a training of a network.
    """

    def __init__(self, params, data_handler):
        """
        Create an ObjectiveBase object.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this objective.

        data_handler : mala.datahandling.data_handler.DataHandler
            datahandler to be used during the hyperparameter optimization.
        """
        self.params = params
        self.data_handler = data_handler

        # We need to find out if we have to reparametrize the lists with the
        # layers and the activations.
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
        """
        Get objective value for a trial (=set of hyperparameters).

        Parameters
        ----------
        trial
            A trial is a set of hyperparameters; can be an optuna based
            trial or simply a OAT compatible list.
        """
        # Parse the parameters included in the trial.
        self.parse_trial(trial)

        # Perform training and report best test loss.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params, test_network, self.data_handler)
        test_trainer.train_network()
        return test_trainer.final_validation_loss

    def parse_trial(self, trial):
        """
        Parse a trial into a network architecture.

        Parameters
        ----------
        trial
            A trial is a set of hyperparameters; can be an optuna based
            trial or simply a OAT compatible list.
        """
        if self.trial_type == "optuna":
            self.parse_trial_optuna(trial)
        elif self.trial_type == "oat":
            self.parse_trial_oat(trial)
        else:
            raise Exception("Cannot parse trial, unknown hyperparameter"
                            " optimization method.")

    def parse_trial_optuna(self, trial: Trial):
        """
        Parse an optuna style trial into the params attribute.

        Parameters
        ----------
        trial : optuna.trial.Trial.
            A set of hyperparameters encoded by optuna.
        """
        if self.optimize_layer_list:
            self.params.network.layer_sizes = \
                [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: HyperparameterOptuna
        for par in self.params.hyperparameters.hlist:
            if par.name == "learning_rate":
                self.params.running.learning_rate = par.get_parameter(trial)
            elif "layer_activation" in par.name:
                self.params.network.layer_activations.\
                    append(par.get_parameter(trial))
            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    self.params.network.layer_sizes.\
                        append(par.get_parameter(trial))
            elif "trainingtype" in par.name:
                self.params.running.trainingtype = par.get_parameter(trial)
            else:
                raise Exception("Optimization of hyperparameter ", par.name,
                                "not supported at the moment.")
        if self.optimize_layer_list:
            self.params.network.layer_sizes.\
                append(self.data_handler.get_output_dimension())

    def parse_trial_oat(self, trial):
        """
        Parse an OA based trial into the params attribute.

        Parameters
        ----------
        trial : numpy.array
            Row in an orthogonal array which respresents current trial.
        """
        if self.optimize_layer_list:
            self.params.network.layer_sizes = \
                [self.data_handler.get_input_dimension()]

        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: HyperparameterOAT
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "learning_rate" in par.name:
                self.params.running.learning_rate = \
                    par.get_parameter(trial, factor_idx)
            elif "layer_activation" in par.name:
                self.params.network.layer_activations.\
                    append(par.get_parameter(trial, factor_idx))
            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    self.params.network.layer_sizes.\
                        append(par.get_parameter(trial, factor_idx))
            elif "trainingtype" in par.name:
                self.params.running.trainingtype = par.\
                    get_parameter(trial, factor_idx)
            else:
                raise Exception("Optimization of hyperparameter ", par.name,
                                "not supported at the moment.")

        if self.optimize_layer_list:
            self.params.network.layer_sizes.\
                append(self.data_handler.get_output_dimension())
