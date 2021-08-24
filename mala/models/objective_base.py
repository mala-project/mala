"""Objective function for all training based hyperparameter optimizations."""
import numpy as np
from optuna import Trial
from .hyperparameter_optuna import HyperparameterOptuna
from .hyperparameter_oat import HyperparameterOAT
from .network import Network
from .trainer import Trainer
from mala import printout


class ObjectiveBase:
    """
    Represents the objective function of a training process.

    This is usually the result of a training of a models.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this objective.

    data_handler : mala.datahandling.data_handler.DataHandler
        datahandler to be used during the hyperparameter optimization.
    """

    def __init__(self, params, data_handler):
        self.params = params
        self.data_handler = data_handler

        # We need to find out if we have to reparametrize the lists with the
        # layers and the activations.
        self.optimize_layer_list = any(map(
            lambda p: "ff_neurons_layer" in p.name,
            self.params.hyperparameters.hlist
        ))
        self.optimize_activation_list = list(map(
            lambda p: "layer_activation" in p.name,
            self.params.hyperparameters.hlist
        )).count(True)

        self.trial_type = self.params.hyperparameters.hyper_opt_method
        if self.trial_type == "notraining":
            self.trial_type = "optuna"

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

        # Train a models for as often as the user desires.
        final_validation_loss = []
        for i in range(0, self.params.hyperparameters.
                number_training_per_trial):
            test_network = Network(self.params)
            test_trainer = Trainer(self.params, test_network, self.data_handler)
            test_trainer.train_model()
            final_validation_loss.append(test_trainer.final_validation_loss)

        if self.params.hyperparameters.number_training_per_trial > 1:
            printout("Losses from multiple runs are: ")
            printout(final_validation_loss)
        return np.mean(final_validation_loss)

    def parse_trial(self, trial):
        """
        Parse a trial into a models architecture.

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
            self.params.model.layer_sizes = \
                [self.data_handler.get_input_dimension()]
        if self.optimize_activation_list > 0:
            self.params.model.layer_activations = []

        # Some layers may have been turned off by optuna.
        turned_off_layers = []

        # This is one because of the input layer.
        layer_counter = 1

        par: HyperparameterOptuna
        for par in self.params.hyperparameters.hlist:
            if par.name == "learning_rate":
                self.params.running.learning_rate = par.get_parameter(trial)

            elif "ff_neurons_layer" in par.name:
                if self.params.model.type == "feed-forward":
                    # Check for zero neuron layers; These indicate layers
                    # that can be left out.
                    layer_size = par.get_parameter(trial)
                    if layer_size > 0:
                        self.params.model.layer_sizes.\
                            append(par.get_parameter(trial))
                    else:
                        turned_off_layers.append(layer_counter)
                    layer_counter += 1

            elif "trainingtype" in par.name:
                self.params.running.trainingtype = par.get_parameter(trial)

            elif "mini_batch_size" in par.name:
                self.params.running.mini_batch_size = par.get_parameter(trial)

            elif "early_stopping_epochs" in par.name:
                self.params.running.early_stopping_epochs = par.\
                    get_parameter(trial)

            elif "learning_rate_patience" in par.name:
                self.params.running.learning_rate_patience = par.\
                    get_parameter(trial)

            elif "learning_rate_decay" in par.name:
                self.params.running.learning_rate_decay = par.\
                    get_parameter(trial)

            elif "layer_activation" in par.name:
                pass

            else:
                raise Exception("Optimization of hyperparameter ", par.name,
                                "not supported at the moment.")

        # We have to process the activations separately, because they depend on
        # the results of the layer lists.

        layer_counter = 0
        for par in self.params.hyperparameters.hlist:
            if "layer_activation" in par.name:
                if layer_counter not in turned_off_layers:
                    self.params.model.layer_activations.\
                        append(par.get_parameter(trial))
                layer_counter += 1

        if self.optimize_layer_list:
            self.params.model.layer_sizes.\
                append(self.data_handler.get_output_dimension())

    def parse_trial_oat(self, trial):
        """
        Parse an OA based trial into the params attribute.

        Parameters
        ----------
        trial : numpy.array
            Row in an orthogonal array which respresents current trial.
        """
        if self.optimize_activation_list > 0:
            self.params.model.layer_activations = []

        par: HyperparameterOAT
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                self.params.model.layer_activations.\
                    append(par.get_parameter(trial, factor_idx))
            elif "trainingtype" in par.name:
                self.params.running.trainingtype = par.\
                    get_parameter(trial, factor_idx)
            else:
                raise Exception("Optimization of hyperparameter ", par.name,
                                "not supported at the moment.")
