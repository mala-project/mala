"""Prunes a network when the score is above a user defined limit."""

import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
from mala import Parameters, DataHandler
from .objective_no_training import ObjectiveNoTraining


class NoTrainingPruner(BasePruner):
    def __init__(self, search_parameters: Parameters, data_handler:
                 DataHandler, trial_type):
        self._trial_type = trial_type
        self._data_handler = data_handler
        self._params = search_parameters

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        objective = ObjectiveNoTraining(self._params, self._data_handler,
                                        self._trial_type)
        surrogate_loss = objective(trial)
        if surrogate_loss > self._params.hyperparameters.no_training_cutoff:
            return True
        else:
            return False
