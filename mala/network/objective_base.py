"""Objective function for all training based hyperparameter optimizations."""
import numpy as np
from optuna import Trial, TrialPruned

from mala.network.hyperparameter_optuna import HyperparameterOptuna
from mala.network.hyperparameter_oat import HyperparameterOAT
from mala.network.network import Network
from mala.network.trainer import Trainer
from mala import printout


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
        contains_single_layer = any(map(
            lambda p: "ff_neurons_layer" in p.name,
            self.params.hyperparameters.hlist
        ))
        contains_multiple_layer_neurons = any(map(
            lambda p: "ff_multiple_layers_neurons" in p.name,
            self.params.hyperparameters.hlist
        ))
        contains_multiple_layers_count = any(map(
            lambda p: "ff_multiple_layers_count" in p.name,
            self.params.hyperparameters.hlist
        ))
        if contains_multiple_layer_neurons != contains_multiple_layers_count:
            print("You selected multiple layers to be optimized, but either "
                  "the range of neurons or number of layers is missing. "
                  "This input will be ignored.")
        self.optimize_layer_list = contains_single_layer or (
                    contains_multiple_layer_neurons and
                    contains_multiple_layers_count)
        self.optimize_activation_list = list(map(
            lambda p: "layer_activation" in p.name,
            self.params.hyperparameters.hlist
        )).count(True)

        self.trial_type = self.params.hyperparameters.hyper_opt_method

    def __call__(self, trial):
        """
        Get objective value for a trial (=set of hyperparameters).

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial is a set of hyperparameters; can be an optuna based
            trial or simply a OAT compatible list.
        """
        # Parse the parameters included in the trial.
        self.parse_trial(trial)
        if self.trial_type == "optuna" and self.params.hyperparameters.pruner\
                == "naswot":
            if trial.should_prune():
                raise TrialPruned()

        # Train a network for as often as the user desires.
        final_validation_loss = []
        for i in range(0, self.params.hyperparameters.
                       number_training_per_trial):
            test_network = Network(self.params)
            test_trainer = Trainer(self.params, test_network,
                                   self.data_handler)
            test_trainer.train_network()
            final_validation_loss.append(test_trainer.final_validation_loss)
            if self.trial_type == "optuna" and \
                    self.params.hyperparameters.pruner \
                    == "multi_training":

                # This is a little bit hacky, since report is actually
                # meant for values DURING training, but we instead
                # use it for one of the losses during multiple trainings.
                # It should not pose a problem though.
                trial.report(test_trainer.final_validation_loss, i)
                if trial.should_prune():
                    raise TrialPruned()

        if self.params.hyperparameters.number_training_per_trial > 1:
            printout("Losses from multiple runs are: ", min_verbosity=2)
            printout(final_validation_loss, min_verbosity=2)

        if self.params.hyperparameters.trial_ensemble_evaluation == "mean":
            return np.mean(final_validation_loss)

        elif self.params.hyperparameters.trial_ensemble_evaluation == \
                "mean_std":
            mean = np.mean(final_validation_loss)

            # Cannot calculate the standar deviation of a bunch of infinities.
            if np.isinf(mean):
                return mean
            else:
                return np.mean(final_validation_loss) + \
                       np.std(final_validation_loss)
        else:
            raise Exception("No way to estimate the trial metric from ensemble"
                            " training provided.")

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
                [self.data_handler.input_dimension]
        if self.optimize_activation_list > 0:
            self.params.network.layer_activations = []

        # Some layers may have been turned off by optuna.
        turned_off_layers = []

        # This is one because of the input layer.
        layer_counter = 1

        par: HyperparameterOptuna
        for par in self.params.hyperparameters.hlist:
            if par.name == "learning_rate":
                self.params.running.learning_rate = par.get_parameter(trial)

            # If the user wants to optimize multiple layers simultaneously,
            # we have to parse to parameters at the same time.
            elif par.name == "ff_multiple_layers_neurons":
                neurons_per_layer = par.get_parameter(trial)
                number_layers = 0
                max_number_layers = 0
                other_par: HyperparameterOptuna
                for other_par in self.params.hyperparameters.hlist:
                    if other_par.name == "ff_multiple_layers_count":
                        number_layers = other_par.get_parameter(trial)
                        max_number_layers = max(other_par.choices)
                if number_layers > 0:
                    for i in range(0, number_layers):
                        if neurons_per_layer > 0:
                            self.params.network.layer_sizes. \
                                append(neurons_per_layer)
                        else:
                            turned_off_layers.append(layer_counter)
                        layer_counter += 1
                    if number_layers != max_number_layers:
                        for i in range(number_layers, max_number_layers):
                            turned_off_layers.append(layer_counter)
                            layer_counter += 1
                else:
                    for i in range(0, max_number_layers):
                        turned_off_layers.append(layer_counter)
                        layer_counter += 1

            elif par.name == "ff_multiple_layers_count":
                # This is parsed directly abve.
                pass

            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    # Check for zero neuron layers; These indicate layers
                    # that can be left out.
                    layer_size = par.get_parameter(trial)
                    if layer_size > 0:
                        self.params.network.layer_sizes.\
                            append(par.get_parameter(trial))
                    else:
                        turned_off_layers.append(layer_counter)
                    layer_counter += 1

            elif "trainingtype" == par.name:
                self.params.running.trainingtype = par.get_parameter(trial)

            elif "mini_batch_size" == par.name:
                self.params.running.mini_batch_size = par.get_parameter(trial)

            elif "early_stopping_epochs" == par.name:
                self.params.running.early_stopping_epochs = par.\
                    get_parameter(trial)

            elif "learning_rate_patience" == par.name:
                self.params.running.learning_rate_patience = par.\
                    get_parameter(trial)

            elif "learning_rate_decay" == par.name:
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
                    self.params.network.layer_activations.\
                        append(par.get_parameter(trial))
                layer_counter += 1

        if self.optimize_layer_list:
            self.params.network.layer_sizes.\
                append(self.data_handler.output_dimension)

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
                [self.data_handler.input_dimension]

        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        # Some layers may have been turned off by optuna.
        turned_off_layers = []

        # This is one because of the input layer.
        layer_counter = 1

        par: HyperparameterOAT
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "learning_rate" == par.name:
                self.params.running.learning_rate = \
                    par.get_parameter(trial, factor_idx)
                # If the user wants to optimize multiple layers simultaneously,
                # we have to parse to parameters at the same time.
            elif par.name == "ff_multiple_layers_neurons":
                neurons_per_layer = par.get_parameter(trial, factor_idx)
                number_layers = 0
                max_number_layers = 0
                other_par: HyperparameterOAT
                for other_idx, other_par in enumerate(self.params.
                                                      hyperparameters.hlist):
                    if other_par.name == "ff_multiple_layers_count":
                        number_layers = other_par.get_parameter(trial,
                                                                other_idx)
                        max_number_layers = max(other_par.choices)
                if number_layers > 0:
                    for i in range(0, number_layers):
                        if neurons_per_layer > 0:
                            self.params.network.layer_sizes. \
                                append(neurons_per_layer)
                        else:
                            turned_off_layers.append(layer_counter)
                        layer_counter += 1
                    if number_layers != max_number_layers:
                        for i in range(number_layers, max_number_layers):
                            turned_off_layers.append(layer_counter)
                            layer_counter += 1
                else:
                    for i in range(0, max_number_layers):
                        turned_off_layers.append(layer_counter)
                        layer_counter += 1

            elif par.name == "ff_multiple_layers_count":
                # This is parsed directly abve.
                pass

            elif "ff_neurons_layer" in par.name:
                if self.params.network.nn_type == "feed-forward":
                    # Check for zero neuron layers; These indicate layers
                    # that can be left out.
                    layer_size = par.get_parameter(trial, factor_idx)
                    if layer_size > 0:
                        self.params.network.layer_sizes. \
                            append(par.get_parameter(trial, factor_idx))
                    else:
                        turned_off_layers.append(layer_counter)
                    layer_counter += 1

            elif "trainingtype" == par.name:
                self.params.running.trainingtype = par.\
                    get_parameter(trial, factor_idx)
            elif "mini_batch_size" == par.name:
                self.params.running.mini_batch_size = \
                    par.get_parameter(trial, factor_idx)
            elif "early_stopping_epochs" == par.name:
                self.params.running.early_stopping_epochs = par.\
                    get_parameter(trial, factor_idx)

            elif "learning_rate_patience" == par.name:
                self.params.running.learning_rate_patience = par.\
                    get_parameter(trial, factor_idx)

            elif "learning_rate_decay" == par.name:
                self.params.running.learning_rate_decay = par.\
                    get_parameter(trial,factor_idx)

            elif "layer_activation" in par.name:
                pass

            else:
                raise Exception("Optimization of hyperparameter ", par.name,
                                "not supported at the moment.")

        # We have to process the activations separately, because they depend on
        # the results of the layer lists.

        layer_counter = 0
        par: HyperparameterOAT
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                if layer_counter not in turned_off_layers:
                    self.params.network.layer_activations.\
                        append(par.get_parameter(trial, factor_idx))
                layer_counter += 1

        if self.optimize_layer_list:
            self.params.network.layer_sizes.\
                append(self.data_handler.output_dimension)
