"""Hyperparameter optimizer using optuna."""
import pickle
import optuna
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase
from mala.common.parameters import printout
from mala.datahandling.data_handler import DataHandler
from mala.datahandling.data_scaler import DataScaler
from mala.common.parameters import Parameters


class HyperOptOptuna(HyperOptBase):
    """Hyperparameter optimizer using Optuna."""

    def __init__(self, params, data, trials_from_last_checkpoint=0):
        """
        Create a HyperOptOptuna object.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this hyperparameter optimizer.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the data for the hyperparameter optimization.

        trials_from_last_checkpoint : int
            Trials that were performed during the last checkpoint.
        """
        super(HyperOptOptuna, self).__init__(params, data)
        self.params = params

        # Make the sample behave in a reproducible way, if so specified by
        # the user.
        sampler = None
        if params.manual_seed is not None:
            sampler = optuna.samplers.TPESampler(seed=params.manual_seed)

        # Create the study.
        if self.params.hyperparameters.rdb_storage is None:
            self.study = optuna.\
                create_study(direction=self.params.hyperparameters.direction,
                             sampler=sampler,
                             study_name=self.params.hyperparameters.
                             study_name)
        else:
            if self.params.hyperparameters.study_name is None:
                raise Exception("If RDB storage is used, a name for the study "
                                "has to be provided.")
            self.study = optuna.\
                create_study(direction=self.params.hyperparameters.direction,
                             sampler=sampler,
                             study_name=self.params.hyperparameters.
                             study_name,
                             storage=self.params.hyperparameters.rdb_storage,
                             load_if_exists=True)

        self.objective = None
        self.checkpoint_counter = 0
        self.trials_from_last_checkpoint = trials_from_last_checkpoint

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, optuna is used.
        """
        self.objective = ObjectiveBase(self.params, self.data_handler)

        # Fill callback list based on user checkpoint wishes.
        callback_list = []
        if self.params.hyperparameters.checkpoints_each_trial != 0:
            callback_list.append(self.__create_checkpointing)

        self.study.optimize(self.objective,
                            n_trials=self.params.hyperparameters.n_trials -
                            self.trials_from_last_checkpoint,
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
        return self.study.get_trials()

    @classmethod
    def resume_checkpoint(cls, checkpoint_name):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Please note that to actually resume the optimization,
        HyperOptOptuna.perform_study() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOptuna
            The hyperparameter optimizer reconstructed from the checkpoint.
        """
        printout("Loading hyperparameter optimization from checkpoint.")
        # The names are based upon the checkpoint name.
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        param_name = checkpoint_name + "_params.pkl"
        optimizer_name = checkpoint_name + "_hyperopt.pth"

        # First load the all the regular objects.
        loaded_params = Parameters.load_from_file(param_name)
        loaded_iscaler = DataScaler.load_from_file(iscaler_name)
        loaded_oscaler = DataScaler.load_from_file(oscaler_name)

        printout("Preparing data used for last checkpoint.")
        # Create a new data handler and prepare the data.
        new_datahandler = DataHandler(loaded_params,
                                      input_data_scaler=loaded_iscaler,
                                      output_data_scaler=loaded_oscaler)
        new_datahandler.prepare_data(reparametrize_scaler=False)
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
        with open(file_path, 'rb') as handle:
            loaded_study = pickle.load(handle)

        # Now, create the Trainer class with it.
        loaded_hyperopt = HyperOptOptuna(params, data,
                                         trials_from_last_checkpoint=
                                         len(loaded_study.get_trials()))
        loaded_hyperopt.study = loaded_study
        return loaded_hyperopt

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
                                             "optimization.")
        if self.params.hyperparameters.checkpoints_each_trial < 0 and \
                trial.number == study.best_trial.number:
            need_to_checkpoint = True
            printout("Best trial is "+str(trial.number)+", creating a "
                     "checkpoint for it.")

        if need_to_checkpoint is True:
            # We need to create a checkpoint!
            self.checkpoint_counter = 0

            # Get the filenames.
            iscaler_name = self.params.hyperparameters.checkpoint_name \
                           + "_iscaler.pkl"
            oscaler_name = self.params.hyperparameters.checkpoint_name \
                           + "_oscaler.pkl"
            param_name = self.params.hyperparameters.checkpoint_name \
                           + "_params.pkl"
            hyperopt_name = self.params.hyperparameters.checkpoint_name \
                           + "_hyperopt.pth"

            # First we save the objects we would also save for inference.
            self.data_handler.input_data_scaler.save(iscaler_name)
            self.data_handler.output_data_scaler.save(oscaler_name)
            self.params.save(param_name)

            # Next, we save all the other objects.
            # Here some horovod stuff would have to go.
            # But so far, the optuna implementation is not horovod-ready...
            # if self.params.use_horovod:
            #     if hvd.rank() != 0:
            #         return

            with open(hyperopt_name, 'wb') as handle:
                pickle.dump(self.study, handle, protocol=4)
