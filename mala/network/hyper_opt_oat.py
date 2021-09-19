"""Hyperparameter optimizer using orthogonal array tuning."""
from bisect import bisect
import itertools
import warnings
import os
import pickle

import numpy as np
try:
    import oapackage as oa
except ModuleNotFoundError:
    warnings.warn("You do not have the OApackage installed. This will not "
                  "affect MALA performance except for when attempting to use "
                  "orthogonal array tuning. ",
                  stacklevel=2)

from mala.network.hyper_opt_base import HyperOptBase
from mala.network.objective_base import ObjectiveBase
from mala.network.hyperparameter_oat import HyperparameterOAT
from mala.common.printout import printout
from mala.common.parameters import Parameters
from mala.datahandling.data_handler import DataHandler
from mala.datahandling.data_scaler import DataScaler


class HyperOptOAT(HyperOptBase):
    """Hyperparameter optimizer using Orthogonal Array Tuning.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.
    """

    def __init__(self, params, data):
        super(HyperOptOAT, self).__init__(params, data)
        self.objective = None
        self.optimal_params = None
        self.checkpoint_counter = 0

        # Related to the OA itself.
        self.importance = None
        self.n_factors = None
        self.factor_levels = None
        self.strength = None
        self.N_runs = None
        self.__OA = None

        # Tracking the trial progress.
        self.sorted_num_choices = []
        self.current_trial = 0
        self.trial_losses = None

    def add_hyperparameter(self, opttype="categorical", name="", choices=None, **kwargs):
        """
        Add hyperparameter such that the hyperparameter list is sorted w.r.t the number of choices.

        Parameters
        ----------
        opttype : string
            Datatype of the hyperparameter. Follows optunas naming convetions.
            Default value - categorical (list)
        """
        if not self.sorted_num_choices:  # if empty
            super(HyperOptOAT, self).add_hyperparameter(
                opttype=opttype, name=name, choices=choices)
            self.sorted_num_choices.append(len(choices))

        else:
            index = bisect(self.sorted_num_choices, len(choices))
            self.sorted_num_choices.insert(index, len(choices))
            self.params.hyperparameters.hlist.insert(
                index, HyperparameterOAT(opttype=opttype, name=name, choices=choices))

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, these are choosen based on an orthogonal array.
        """
        self.__OA = self.get_orthogonal_array()
        if self.trial_losses is None:
            self.trial_losses = np.zeros(self.__OA.shape[0])+float("inf")

        printout("Performing",self.N_runs,
                 "trials, starting with trial number", self.current_trial)

        # The parameters could have changed.
        self.objective = ObjectiveBase(self.params, self.data_handler)

        # Iterate over the OA and perform the trials.
        for i in range(self.current_trial, self.N_runs):
            row = self.__OA[i]
            self.trial_losses[self.current_trial] = self.objective(row)

            # Output diagnostic information.
            best_trial = self.get_best_trial_results()
            printout("Trial number", self.current_trial,
                     "finished with:", self.trial_losses[self.current_trial],
                     ", best is trial", best_trial[0],
                     "with", best_trial[1])
            self.current_trial += 1
            self.__create_checkpointing(row)

        # Perform Range Analysis
        self.get_optimal_parameters()

    def get_optimal_parameters(self):
        """
        Find the optimal set of hyperparameters by doing range analysis.

        This is done using loss instead of accuracy as done in the paper.
        """
        printout("Performing Range Analysis.")

        def indices(idx, val): return np.where(
            self.__OA[:, idx] == val)[0]
        R = [[self.trial_losses[indices(idx, l)].sum() for l in range(levels)]
             for (idx, levels) in enumerate(self.factor_levels)]

        A = [[i/len(j) for i in j] for j in R]

        # Taking loss as objective to minimise
        self.optimal_params = np.array([i.index(min(i)) for i in A])
        self.importance = np.argsort([max(i)-min(i) for i in A])

    def show_order_of_importance(self):
        """Print the order of importance of the hyperparameters that are being optimised."""
        printout("Order of Importance: ")
        printout(
            *[self.params.hyperparameters.hlist[idx].name for idx in self.importance], sep=" < ")

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        self.objective.parse_trial_oat(self.optimal_params)

    def get_orthogonal_array(self):
        """
        Generate the best Orthogonal array used for optimal hyperparameter sampling.

        Parameters
        ----------
        factor_levels : list
            A list of number of choices of each hyperparameter

        strength : int
            A design parameter for Orthogonal arrays
                strength 2 models all 2 factor interactions
                strength 3 models all 3 factor interactions

        N_runs : int 
            Minimum number of experimental runs to be performed 

        This is function is taken from the example notebook of OApackage
        """
        self.__check_factor_levels()
        print(self.sorted_num_choices)
        self.n_factors = len(self.params.hyperparameters.hlist)

        self.factor_levels = [par.num_choices for par in self.params.
                              hyperparameters.hlist]

        self.strength = 2
        self.N_runs = self.number_of_runs()
        print("Generating Suitable Orthogonal Array.")
        arrayclass = oa.arraydata_t(self.factor_levels, self.N_runs, self.strength,
                                    self.n_factors)
        arraylist = [arrayclass.create_root()]

        # extending the orthogonal array
        options = oa.OAextend()
        options.setAlgorithmAuto(arrayclass)

        for _ in range(self.strength + 1, self.n_factors + 1):
            arraylist_extensions = oa.extend_arraylist(arraylist, arrayclass,
                                                        options)
            dd = np.array([a.Defficiency() for a in arraylist_extensions])
            idxs = np.argsort(dd)
            arraylist = [arraylist_extensions[ii] for ii in idxs]

        if not arraylist:
            raise Exception("No orthogonal array exists with such a parameter combination.")
            
        else:
            return np.unique(np.array(arraylist[0]), axis=0)

    def number_of_runs(self):
        """
        Calculate the minimum number of runs required for an Orthogonal array.

        Based on the factor levels and the strength of the array requested
        """
        runs = [np.prod(tt) for tt in itertools.combinations(
            self.factor_levels, self.strength)]

        N = np.lcm.reduce(runs)*np.lcm.reduce(self.factor_levels)
        return int(N)

    def get_best_trial_results(self):
        """Get the best trial out of the list, including the value."""
        if self.params.hyperparameters.direction == "minimize":
            return [np.argmin(self.trial_losses), np.min(self.trial_losses)]
        elif self.params.hyperparameters.direction == "maximize":
            return [np.argmax(self.trial_losses), np.max(self.trial_losses)]
        else:
            raise Exception("Invalid direction for hyperparameter optimization"
                            "selected.")

    def __check_factor_levels(self):
        """Checks that the factors are in a decreasing order."""
        dx = np.diff(self.sorted_num_choices)
        if np.all(dx >= 0):
            # Factors in increasing order, we have to reverse the order.
            self.sorted_num_choices.reverse()
            self.params.hyperparameters.hlist.reverse()
        elif np.all(dx <= 0):
            # Factors are in decreasing order, we don't have to do anything.
            pass
        else:
            raise Exception("Please use hyperparameters in increasing or "
                            "decreasing order of number of choices")

    @classmethod
    def checkpoint_exists(cls, checkpoint_name):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        param_name = checkpoint_name + "_params.pkl"

        return all(map(os.path.isfile, [iscaler_name, oscaler_name,
                                        param_name]))

    @classmethod
    def resume_checkpoint(cls, checkpoint_name,
                          no_data=False):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Please note that to actually resume the optimization,
        HyperOptOptuna.perform_study() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which the checkpoint is loaded.

        no_data : bool
            If True, the data won't actually be loaded into RAM or scaled.
            This can be useful for cases where a checkpoint is loaded
            for analysis purposes.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOAT
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
        if no_data is True:
            loaded_params.data.use_lazy_loading = True
        new_datahandler = DataHandler(loaded_params,
                                      input_data_scaler=loaded_iscaler,
                                      output_data_scaler=loaded_oscaler)
        new_datahandler.prepare_data(reparametrize_scaler=False)
        new_hyperopt = HyperOptOAT.load_from_file(loaded_params,
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
        loaded_hyperopt : HyperOptOAT
            The hyperparameter optimizer that was loaded from the file.
        """
        # First, load the checkpoint.
        with open(file_path, 'rb') as handle:
            loaded_tracking_data = pickle.load(handle)
            loaded_hyperopt = HyperOptOAT(params, data)
            loaded_hyperopt.sorted_num_choices = loaded_tracking_data[0]
            loaded_hyperopt.current_trial = loaded_tracking_data[1]
            loaded_hyperopt.trial_losses = loaded_tracking_data[2]

        return loaded_hyperopt

    def __create_checkpointing(self, trial):
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
                np.argmin(self.trial_losses) == self.current_trial-1:
            need_to_checkpoint = True
            printout("Best trial is "+str(self.current_trial-1)+", creating a "
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
            # The study only has to be saved if the no RDB storage is used.
            if self.params.hyperparameters.rdb_storage is None:
                hyperopt_name = self.params.hyperparameters.checkpoint_name \
                            + "_hyperopt.pth"
                study = [self.sorted_num_choices, self.current_trial,
                         self.trial_losses]
                with open(hyperopt_name, 'wb') as handle:
                    pickle.dump(study, handle, protocol=4)
