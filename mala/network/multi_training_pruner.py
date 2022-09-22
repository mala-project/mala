"""Prunes a trial when one of the trainings returns infinite band energy."""
import numpy as np
import optuna
from optuna.pruners import BasePruner

from mala import Parameters
from mala.common.parallelizer import parallel_warn


class MultiTrainingPruner(BasePruner):
    """
    Implements the NASWOT method (first version of paper) as an optuna pruner.

    This means that before each training of a trial network the network will
    be tested against a user defined surrogate score (which has to be
    calibrated). If this score is good enough, the candidate will be trained.

    Parameters
    ----------
    search_parameters : mala.common.parametes.Parameters
        Parameters used to create this objective.
    """

    def __init__(self, search_parameters: Parameters):
        self._params = search_parameters
        self._trial_type = self._params.hyperparameters.hyper_opt_method
        if self._trial_type != "optuna":
            raise Exception("This pruner only works for optuna at the moment.")
        if self._params.hyperparameters.number_training_per_trial == 1:
            parallel_warn("This pruner has no effect if only one training per "
                          "trial is performed.")

    def prune(self, study: "optuna.study.Study",
              trial: "optuna.trial.FrozenTrial") -> bool:
        """
        Judge whether the trial should be pruned based on the reported values.

        In this case, it is checked whether one of the multiple trainings
        reported an infinte loss (especially viable for the band_energy
        metric).

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
        # We always overwrite the first loss.
        last_loss = trial.intermediate_values[trial.last_step]
        if np.isinf(last_loss):
            return True
        else:
            return False
