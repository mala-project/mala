"""Prunes a network when the score is above a user defined limit."""
import optuna
from optuna.pruners import BasePruner

from mala import Parameters, DataHandler
from mala.network.objective_naswot import ObjectiveNASWOT


class NASWOTPruner(BasePruner):
    """
    Implements the NASWOT method (first version of paper) as an optuna pruner.

    This means that before each training of a trial network the network will
    be tested against a user defined surrogate score (which has to be
    calibrated). If this score is good enough, the candidate will be trained.

    Parameters
    ----------
    search_parameters : mala.common.parametes.Parameters
        Parameters used to create this objective.

    data_handler : mala.datahandling.data_handler.DataHandler
        datahandler to be used during the hyperparameter optimization.

    """

    def __init__(self, search_parameters: Parameters, data_handler:
                 DataHandler):
        self._data_handler = data_handler
        self._params = search_parameters
        self._trial_type = self._params.hyperparameters.hyper_opt_method
        if self._trial_type != "optuna":
            raise Exception("This pruner only works for optuna at the moment.")

    def prune(self, study: "optuna.study.Study", trial:
              "optuna.trial.FrozenTrial") -> bool:
        """
        Judge whether the trial should be pruned based on the reported values.

        Note that this method is not supposed to be called by library users. 
        Instead, :func:`optuna.trial.Trial.report` and 
        :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective 
        function.
        
        Parameters
        ----------
        study : optuna.study.Study
            Study object of the target study.

        trial : optuna.trial.FrozenTrial
            FrozenTrial object of the target trial.
            Take a copy before modifying this object.

        Returns
        -------
        should_prune : bool 
            A boolean indicating whether this particular trial should be 
            pruned. 
        """
        objective = ObjectiveNASWOT(self._params, self._data_handler,
                                    self._trial_type, batch_size=
                                        self._params.hyperparameters.
                                    naswot_pruner_batch_size)
        surrogate_loss = objective(trial)
        if surrogate_loss < self._params.hyperparameters.naswot_pruner_cutoff:
            return True
        else:
            return False
