"""Hyperparameter optimizer using orthogonal array tuning."""

from bisect import bisect
import itertools
import pickle

import numpy as np

try:
    import oapackage as oa
except ModuleNotFoundError:
    pass

from mala.network.hyper_opt import HyperOpt
from mala.network.objective_base import ObjectiveBase
from mala.network.hyperparameter_oat import HyperparameterOAT
from mala.common.parallelizer import printout, parallel_warn


class HyperOptOAT(HyperOpt):
    """Hyperparameter optimizer using Orthogonal Array Tuning.

    Based on https://link.springer.com/chapter/10.1007/978-3-030-36808-1_31.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, data, use_pkl_checkpoints=False):
        super(HyperOptOAT, self).__init__(
            params, data, use_pkl_checkpoints=use_pkl_checkpoints
        )
        self._objective = None
        self._optimal_params = None
        self._checkpoint_counter = 0

        # Related to the OA itself.
        self._importance = None
        self._n_factors = None
        self._factor_levels = None
        self._strength = None
        self._N_runs = None
        self.__OA = None

        # Tracking the trial progress.
        self._sorted_num_choices = []
        self._current_trial = 0
        self._trial_losses = None

    @property
    def best_trial_index(self):
        """
        Get the index and loss of best trial determined in this NASWOT run.

        This property is read only, and will be recomputed.

        Returns
        -------
        best_trial_index : list
            A list containing [0] the best trial index and [1] the best
            trial loss.
        """
        if self._trial_losses is None:
            parallel_warn(
                "Trial list is not yet computed, cannot determine "
                "best trial."
            )
            return [-1, np.inf]

        if self.params.hyperparameters.direction == "minimize":
            return [np.argmin(self._trial_losses), np.min(self._trial_losses)]
        elif self.params.hyperparameters.direction == "maximize":
            return [np.argmax(self._trial_losses), np.max(self._trial_losses)]
        else:
            raise Exception(
                "Invalid direction for hyperparameter optimization selected."
            )

    @best_trial_index.setter
    def best_trial_index(self, value):
        pass

    def add_hyperparameter(
        self, opttype="categorical", name="", choices=None, **kwargs
    ):
        """
        Add hyperparameter.

        Hyperparameter list will automatically sorted w.r.t the number of
        choices.

        Parameters
        ----------
        opttype : string
            Datatype of the hyperparameter. Follows optuna's naming
            conventions, but currently only supports "categorical" (a list).
        """
        if not self._sorted_num_choices:  # if empty
            super(HyperOptOAT, self).add_hyperparameter(
                opttype=opttype, name=name, choices=choices
            )
            self._sorted_num_choices.append(len(choices))

        else:
            index = bisect(self._sorted_num_choices, len(choices))
            self._sorted_num_choices.insert(index, len(choices))
            self.params.hyperparameters.hlist.insert(
                index,
                HyperparameterOAT(opttype=opttype, name=name, choices=choices),
            )

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        Internally constructs an orthogonal array and performs trial NN
        trainings based on it.
        """
        if self.__OA is None:
            self.__OA = self._get_orthogonal_array()
        print(self.__OA)
        if self._trial_losses is None:
            self._trial_losses = np.zeros(self.__OA.shape[0]) + float("inf")

        printout(
            "Performing",
            self._N_runs,
            "trials, starting with trial number",
            self._current_trial,
            min_verbosity=0,
        )

        # The parameters could have changed.
        self._objective = ObjectiveBase(self.params, self._data_handler)

        # Iterate over the OA and perform the trials.
        for i in range(self._current_trial, self._N_runs):
            row = self.__OA[i]
            self._trial_losses[self._current_trial] = self._objective(row)

            # Output diagnostic information.
            printout(
                "Trial number",
                self._current_trial,
                "finished with:",
                self._trial_losses[self._current_trial],
                ", best is trial",
                self.best_trial_index[0],
                "with",
                self.best_trial_index[1],
                min_verbosity=0,
            )
            self._current_trial += 1
            self.__create_checkpointing(row)

        # Perform Range Analysis
        self._range_analysis()

    def _range_analysis(self):
        """
        Find the optimal set of hyperparameters by doing range analysis.

        This is done using loss instead of accuracy as done in the paper.
        """
        printout("Performing Range Analysis.", min_verbosity=1)

        def indices(idx, val):
            return np.where(self.__OA[:, idx] == val)[0]

        R = [
            [self._trial_losses[indices(idx, l)].sum() for l in range(levels)]
            for (idx, levels) in enumerate(self._factor_levels)
        ]

        A = [[i / len(j) for i in j] for j in R]

        # Taking loss as objective to minimise
        self._optimal_params = np.array([i.index(min(i)) for i in A])
        self._importance = np.argsort([max(i) - min(i) for i in A])

    def show_order_of_importance(self):
        """Print the order of importance of the hyperparameters."""
        printout("Order of Importance: ", min_verbosity=0)
        printout(
            *[
                self.params.hyperparameters.hlist[idx].name
                for idx in self._importance
            ],
            sep=" < ",
            min_verbosity=0
        )

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        self._objective.parse_trial_oat(self._optimal_params)

    def _get_orthogonal_array(self):
        """
        Generate the best OA used for optimal hyperparameter sampling.

        This is function is taken from the example notebook of OApackage.

        Returns
        -------
        orthogonal_array : numpy.ndarray
            The orthogonal array used for hyperparameter optimization.
        """
        self.__check_factor_levels()
        print("Sorted factor levels:", self._sorted_num_choices)
        self._n_factors = len(self.params.hyperparameters.hlist)

        self._factor_levels = [
            par.num_choices for par in self.params.hyperparameters.hlist
        ]

        self._strength = 2
        arraylist = None

        # This is a little bit hacky.
        # What happens is that while we can _technically_ evaluate N_runs
        # analytically, depending on the actual factor levels, such an array
        # might not exist. We know however, that one exists with N_runs_actual
        # for which:
        # N_runs_actual = N_runs_analytical * x
        # holds. x is unknown, but we can be confident that it should be
        # small. So simply trying 3 time should be fine for now.
        for i in range(1, 4):
            self._N_runs = self._number_of_runs() * i
            print("Trying run size:", self._N_runs)
            print("Generating Suitable Orthogonal Array.")
            arrayclass = oa.arraydata_t(
                self._factor_levels,
                self._N_runs,
                self._strength,
                self._n_factors,
            )
            arraylist = [arrayclass.create_root()]

            # extending the orthogonal array
            options = oa.OAextend()
            options.setAlgorithmAuto(arrayclass)

            for _ in range(self._strength + 1, self._n_factors + 1):
                arraylist_extensions = oa.extend_arraylist(
                    arraylist, arrayclass, options
                )
                dd = np.array([a.Defficiency() for a in arraylist_extensions])
                idxs = np.argsort(dd)
                arraylist = [arraylist_extensions[ii] for ii in idxs]
            if arraylist:
                break

        if not arraylist:
            raise Exception(
                "No orthogonal array exists with such a "
                "parameter combination."
            )

        else:
            return np.unique(np.array(arraylist[0]), axis=0)

    def _number_of_runs(self):
        """
        Calculate the minimum number of runs required for an Orthogonal array.

        Based on the factor levels and the strength of the array requested.
        See also here:
        https://oapackage.readthedocs.io/en/latest/examples/example_minimal_number_of_runs_oa.html

        Returns
        -------
        number_of_runs : int
            The number of runs required to sample the orthogonal array.
        """
        runs = [
            np.prod(tt)
            for tt in itertools.combinations(
                self._factor_levels, self._strength
            )
        ]

        N = np.lcm.reduce(runs)
        return int(N)

    def __check_factor_levels(self):
        """Check that the factors are in a decreasing order."""
        dx = np.diff(self._sorted_num_choices)
        if np.all(dx >= 0):
            # Factors in increasing order, we have to reverse the order.
            self._sorted_num_choices.reverse()
            self.params.hyperparameters.hlist.reverse()
        elif np.all(dx <= 0):
            # Factors are in decreasing order, we don't have to do anything.
            pass
        else:
            raise Exception(
                "Please use hyperparameters in increasing or "
                "decreasing order of number of choices"
            )

    @classmethod
    def resume_checkpoint(
        cls, checkpoint_name, no_data=False, use_pkl_checkpoints=False
    ):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Please note that to actually resume the optimization,
        HyperOptOAT.perform_study() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which the checkpoint is loaded.

        no_data : bool
            If True, the data won't actually be loaded into RAM or scaled.
            This can be useful for cases where a checkpoint is loaded
            for analysis purposes.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOAT
            The hyperparameter optimizer reconstructed from the checkpoint.
        """
        loaded_params, new_datahandler, optimizer_name = (
            cls._resume_checkpoint(
                checkpoint_name,
                no_data=no_data,
                use_pkl_checkpoints=use_pkl_checkpoints,
            )
        )
        new_hyperopt = HyperOptOAT.load_from_file(
            loaded_params, optimizer_name, new_datahandler
        )

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
        with open(file_path, "rb") as handle:
            loaded_tracking_data = pickle.load(handle)
            loaded_hyperopt = HyperOptOAT(params, data)
            loaded_hyperopt._sorted_num_choices = loaded_tracking_data[
                "sorted_num_choices"
            ]
            loaded_hyperopt._current_trial = loaded_tracking_data[
                "current_trial"
            ]
            loaded_hyperopt._trial_losses = loaded_tracking_data[
                "trial_losses"
            ]
            loaded_hyperopt._importance = loaded_tracking_data["importance"]
            loaded_hyperopt._n_factors = loaded_tracking_data["n_factors"]
            loaded_hyperopt._factor_levels = loaded_tracking_data[
                "factor_levels"
            ]
            loaded_hyperopt._strength = loaded_tracking_data["strength"]
            loaded_hyperopt._N_runs = loaded_tracking_data["N_runs"]
            loaded_hyperopt.__OA = loaded_tracking_data["OA"]

        return loaded_hyperopt

    def __create_checkpointing(self, trial):
        """Create a checkpoint of optuna study, if necessary."""
        self._checkpoint_counter += 1
        need_to_checkpoint = False

        if (
            self._checkpoint_counter
            >= self.params.hyperparameters.checkpoints_each_trial
            and self.params.hyperparameters.checkpoints_each_trial > 0
        ):
            need_to_checkpoint = True
            printout(
                str(self.params.hyperparameters.checkpoints_each_trial)
                + " trials have passed, creating a "
                "checkpoint for hyperparameter "
                "optimization.",
                min_verbosity=1,
            )
        if (
            self.params.hyperparameters.checkpoints_each_trial < 0
            and np.argmin(self._trial_losses) == self._current_trial - 1
        ):
            need_to_checkpoint = True
            printout(
                "Best trial is "
                + str(self._current_trial - 1)
                + ", creating a "
                "checkpoint for it.",
                min_verbosity=1,
            )

        if need_to_checkpoint is True:
            # We need to create a checkpoint!
            self._checkpoint_counter = 0

            self._save_params_and_scaler()

            # The study only has to be saved if the no RDB storage is used.
            if self.params.hyperparameters.rdb_storage is None:
                hyperopt_name = (
                    self.params.hyperparameters.checkpoint_name
                    + "_hyperopt.pth"
                )

                study = {
                    "sorted_num_choices": self._sorted_num_choices,
                    "current_trial": self._current_trial,
                    "trial_losses": self._trial_losses,
                    "importance": self._importance,
                    "n_factors": self._n_factors,
                    "factor_levels": self._factor_levels,
                    "strength": self._strength,
                    "N_runs": self._N_runs,
                    "OA": self.__OA,
                }
                with open(hyperopt_name, "wb") as handle:
                    pickle.dump(study, handle, protocol=4)
