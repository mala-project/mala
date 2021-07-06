"""Hyperparameter optimizer working without training."""
import optuna

from .hyper_opt_base import HyperOptBase
from .objective_no_training import ObjectiveNoTraining


class HyperOptNoTraining(HyperOptBase):
    """
    Hyperparameter optimizer that does not require training networks.

    Networks are analysed using the Jacobian.
    """

    def __init__(self, params, data):
        """
        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this hyperparameter optimizer.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the data for the hyperparameter optimization.
        """
        super(HyperOptNoTraining, self).__init__(params, data)
        self.objective = None
        self.trial_losses = None
        self.best_trial = None
        self.trial_list = None

    def perform_study(self, trial_list=None):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        Currently it is mandatory to provide a trial_list, although it
        will be optional later on.

        Parameters
        ----------
        trial_list : list
            A list containing trials from either HyperOptOptuna or HyperOptOAT.
            HyperOptNoTraining does currently not have an algorithm to
            create network archtitectures of interestes by itself and insteads
            investigates those sampled by a different hyperparameter
            optimizer.
        """
        # Ideally, this type of HO is called with a list of trials for which
        # the parameter has to be identified.
        self.trial_list = trial_list
        if self.trial_list is None:
            raise Exception("Sorry, Hyperparameter optimization without "
                            "training currently only works if a list of "
                            "trials is provided.")

        # TODO: For now. Needs some refinements later.
        if isinstance(self.trial_list[0], optuna.trial.FrozenTrial):
            trial_type = "optuna"
        else:
            trial_type = "oat"
        self.objective = ObjectiveNoTraining(self.params, self.data_handler,
                                             trial_type)
        self.trial_losses = [self.objective(row) for row in self.trial_list]

        # Return the best lost value we could achieve.
        return min(self.trial_losses)

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        # Getting the best trial based on the test errors
        idx = self.trial_losses.index(min(self.trial_losses))
        self.best_trial = self.trial_list[idx]
        self.objective.parse_trial(self.best_trial)

