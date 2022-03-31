"""Hyperparameter optimizer using optuna."""
import os
import pickle

import optuna

from mala.common.parameters import printout
from mala.network.hyper_opt import HyperOpt
from mala.network.objective_base import ObjectiveBase
from mala.network.naswot_pruner import NASWOTPruner


class HyperOptOptuna(HyperOpt):
    """Hyperparameter optimizer using Optuna.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, data, use_pkl_checkpoints=False):
        super(HyperOptOptuna, self).__init__(params, data,
                                             use_pkl_checkpoints=use_pkl_checkpoints)
        self.params = params

        # Make the sample behave in a reproducible way, if so specified by
        # the user.
        sampler = optuna.samplers.TPESampler(seed=params.manual_seed,
                                             multivariate=params. \
                                             hyperparameters.use_multivariate)

        # See if the user specified a pruner.
        pruner = None
        if self.params.hyperparameters.pruner == "naswot":
            pruner = NASWOTPruner(self.params, data)

        # Create the study.
        if self.params.hyperparameters.rdb_storage is None:
            self.study = optuna.\
                create_study(direction=self.params.hyperparameters.direction,
                             sampler=sampler,
                             study_name=self.params.hyperparameters.
                             study_name,
                             pruner=pruner)
        else:
            if self.params.hyperparameters.study_name is None:
                raise Exception("If RDB storage is used, a name for the study "
                                "has to be provided.")
            rdb_storage = optuna.storages.RDBStorage(
                    url=self.params.hyperparameters.rdb_storage,
                    heartbeat_interval=self.params.hyperparameters.
                    rdb_storage_heartbeat)

            self.study = optuna.\
                create_study(direction=self.params.hyperparameters.direction,
                             sampler=sampler,
                             study_name=self.params.hyperparameters.
                             study_name,
                             storage=rdb_storage,
                             load_if_exists=True,
                             pruner=pruner)
        self.checkpoint_counter = 0

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, optuna is used.
        """
        # The parameters could have changed.
        self.objective = ObjectiveBase(self.params, self.data_handler)

        # Fill callback list based on user checkpoint wishes.
        callback_list = [self.__check_max_number_trials]
        if self.params.hyperparameters.checkpoints_each_trial != 0:
            callback_list.append(self.__create_checkpointing)

        self.study.optimize(self.objective,
                            n_trials=None,
                            callbacks=callback_list)

        # Return the best lost value we could achieve.
        return self.study.best_value

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        # Parse the parameters from the best trial.
        self.objective.parse_trial_optuna(self.study.best_trial)

    def get_trials_from_study(self):
        """
        Return the trials from the last study.

        Returns
        -------
        last_trials: list
            A list of optuna.FrozenTrial objects.
        """
        return self.study.get_trials(states=(optuna.trial.
                                              TrialState.COMPLETE, ))

    @classmethod
    def resume_checkpoint(cls, checkpoint_name, alternative_storage_path=None,
                          no_data=False, use_pkl_checkpoints=False):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Please note that to actually resume the optimization,
        HyperOptOptuna.perform_study() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which the checkpoint is loaded.

        alternative_storage_path: string
            Alternative storage string to load the study from.
            For applications on an HPC cluster it might be necessary to
            slightly modify the storage path between runs, since the SQL
            server might be running on different nodes each time.

        no_data : bool
            If True, the data won't actually be loaded into RAM or scaled.
            This can be useful for cases where a checkpoint is loaded
            for analysis purposes.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOptuna
            The hyperparameter optimizer reconstructed from the checkpoint.
        """
        loaded_params, new_datahandler, optimizer_name = \
            cls._resume_checkpoint(checkpoint_name, no_data=no_data,
                                   use_pkl_checkpoints=use_pkl_checkpoints)
        if alternative_storage_path is not None:
            loaded_params.hyperparameters.rdb_storage = \
                alternative_storage_path
        new_hyperopt = HyperOptOptuna.load_from_file(loaded_params,
                                                     optimizer_name,
                                                     new_datahandler)

        return loaded_params, new_datahandler, new_hyperopt

    @classmethod
    def load_from_file(cls, params, file_path, data):
        """
        Load a hyperparameter optimizer from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the hyperparameter optimizer
            should be created Has to be compatible with data.

        file_path : string
            Path to the file from which the hyperparameter optimizer should
            be loaded.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.

        Returns
        -------
        loaded_trainer : Network
            The hyperparameter optimizer that was loaded from the file.
        """
        # First, load the checkpoint.
        if params.hyperparameters.rdb_storage is None:
            with open(file_path, 'rb') as handle:
                loaded_study = pickle.load(handle)

            # Now, create the Trainer class with it.
            loaded_hyperopt = HyperOptOptuna(params, data)
            loaded_hyperopt.study = loaded_study
        else:
            loaded_hyperopt = HyperOptOptuna(params, data)

        return loaded_hyperopt

    def __get_number_of_completed_trials(self, study):
        """Get the number of maximum trials."""
        # How to calculate this depends on whether or not a heartbeat was
        # used. If one was used, then both COMPLETE and RUNNING trials
        # Can be taken into account, as it can be expected that RUNNING
        # trials will actually finish. If no heartbeat is used,
        # then RUNNING trials might be Zombie trials.
        # See
        if self.params.hyperparameters.rdb_storage_heartbeat is None:
            return len([t for t in study.trials if
                        t.state == optuna.trial.
                        TrialState.COMPLETE])
        else:
            return len([t for t in study.trials if
                        t.state == optuna.trial.
                        TrialState.COMPLETE or
                        t.state == optuna.trial.
                        TrialState.RUNNING])

    def __check_max_number_trials(self, study, trial):
        """Check if this trial was already the maximum number of trials."""
        # How to check for this depends on whether or not a heartbeat was
        # used. If one was used, then both COMPLETE and RUNNING trials
        # Can be taken into account, as it can be expected that RUNNING
        # trials will actually finish. If no heartbeat is used,
        # then RUNNING trials might be Zombie trials.
        # See
        # https://github.com/optuna/optuna/issues/1883#issuecomment-841844834
        # https://github.com/optuna/optuna/issues/1883#issuecomment-842106950

        if self.__get_number_of_completed_trials(study) >= \
                self.params.hyperparameters.n_trials:
            self.study.stop()

    def __create_checkpointing(self, study, trial):
        """Create a checkpoint of optuna study, if necessary."""
        self.checkpoint_counter += 1
        need_to_checkpoint = False

        if self.checkpoint_counter >= self.params.hyperparameters.\
                checkpoints_each_trial and self.params.hyperparameters.\
                checkpoints_each_trial > 0:
            need_to_checkpoint = True
            printout(str(self.params.hyperparameters.
                     checkpoints_each_trial)+" trials have passed, creating a "
                                             "checkpoint for hyperparameter "
                                             "optimization.", min_verbosity=0)
        if self.params.hyperparameters.checkpoints_each_trial < 0 and \
           self.__get_number_of_completed_trials(study) > 0:
                if trial.number == study.best_trial.number:
                    need_to_checkpoint = True
                    printout("Best trial is "+str(trial.number)+", creating a "
                             "checkpoint for it.", min_verbosity=0)

        if need_to_checkpoint is True:
            # We need to create a checkpoint!
            self.checkpoint_counter = 0

            self._save_params_and_scaler()

            # Next, we save all the other objects.
            # Here some horovod stuff would have to go.
            # But so far, the optuna implementation is not horovod-ready...
            # if self.params.use_horovod:
            #     if hvd.rank() != 0:
            #         return
            # The study only has to be saved if the no RDB storage is used.
            if self.params.hyperparameters.rdb_storage is None:
                hyperopt_name = self.params.hyperparameters.checkpoint_name \
                            + "_hyperopt.pth"
                with open(hyperopt_name, 'wb') as handle:
                    pickle.dump(self.study, handle, protocol=4)
