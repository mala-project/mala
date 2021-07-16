"""Hyperparameter optimizer using orthogonal array tuning."""
import oapackage as oa
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase
from .hyperparameter_oat import HyperparameterOAT
import numpy as np
import itertools
from mala.common.printout import printout
from bisect import bisect


class HyperOptOAT(HyperOptBase):
    """Hyperparameter optimizer using Orthogonal Array Tuning."""

    def __init__(self, params, data):
        """
        Create a HyperOptOAT object.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this hyperparameter optimizer.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the data for the hyperparameter optimization.
        """
        super(HyperOptOAT, self).__init__(params, data)
        self.objective = None
        self.trial_losses = []
        self.sorted_num_choices = []
        self.optimal_params = None
        self.importance = None
        self.n_factors = None
        self.factor_levels = None
        self.strength = None
        self.N_runs = None
        self.OA = None

    def add_hyperparameter(self, opttype="categorical", name="", low=0, high=0,
                           choices=None):

        if opttype != 'categorical':
            raise Exception(
                "Only categorical hyperparameters are supported for OAT")

        else:
            if not self.sorted_num_choices:
                super(HyperOptOAT, self).add_hyperparameter(opttype=opttype, name=name,
                                                            low=low, high=high,
                                                            choices=choices)
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
        self.n_factors = len(self.params.hyperparameters.hlist)

        self.factor_levels = [par.num_choices for par in self.params.
                              hyperparameters.hlist]

        printout(*self.factor_levels, sep=",")
        if not self.monotonic:
            raise Exception(
                "Please use hyperparameters in increasing or decreasing order of number of choices")

        self.strength = 2
        self.N_runs = self.number_of_runs()
        self.OA = self.get_orthogonal_array()
        number_of_trial = 0
        self.objective = ObjectiveBase(self.params, self.data_handler)
        for row in self.OA:
            printout("Trial number", number_of_trial)
            self.trial_losses.append(self.objective(row))

            number_of_trial += 1
        self.trial_losses = np.array(self.trial_losses)

        # Return the best loss value we could achieve.
        self.get_optimal_parameters()
        return self.objective(self.optimal_params)

    def get_optimal_parameters(self):
        """
        Find the optimal set of hyperparameters by doing range analysis.
        This is done using loss instead of accuracy as done in the paper.

        """
        printout("Performing Range Analysis")
        print("Factor levels:", self.factor_levels)

        def indices(idx, val): return np.where(
            self.OA[:, idx] == val)[0]
        R = [[self.trial_losses[indices(idx, l)].sum() for l in range(levels)]
             for (idx, levels) in enumerate(self.factor_levels)]

        A = [[i/len(j) for i in j] for j in R]

        print("OA:")
        print(self.OA)

        print("Trial Losses:")
        print(np.array(self.trial_losses).transpose())

        print("R matrix: ")
        print(np.array(R).transpose())

        print("A Matrix:")
        print(np.array(A).transpose())
        # Taking loss as objective to minimise
        self.optimal_params = np.array([i.index(min(i)) for i in A])
        self.importance = np.argsort([max(i)-min(i) for i in A])[::-1]
        printout("Order of Importance: ")
        printout(
            *[self.params.hyperparameters.hlist[idx].name for idx in self.importance], sep=" > ")

        printout("Optimal Hyperparameters:")
        for (idx, par) in enumerate(self.params.hyperparameters.hlist):
            printout(
                par.name, par.choices[self.optimal_params[idx]], sep=' : ')

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """

        self.objective.parse_trial_oat(self.optimal_params)

    def number_of_runs(self):
        """
        Calculate the minimum number of runs required for an Orthogonal array

        Based on the factor levels and the strength of the array requested

        Parameters
        ----------
        factor_levels : list
            A list of number of choices of each hyperparameter

        strength : int
            A design parameter for Orthogonal arrays
                strength 2 models all 2 factor interactions
                strength 3 models all 3 factor interactions

        This is function is taken from the example notebook of OApackage
        """

        runs = [np.prod(tt) for tt in itertools.combinations(
            self.factor_levels, self.strength)]

        N = np.lcm.reduce(runs)
        return int(N)

    def get_orthogonal_array(self):
        """Generate the best Orthogonal array used for optimal hyperparameter sampling."""

        print("Generating Suitable Orthogonal Array")
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

        if not arraylist:  # checking if the list is empty
            raise Exception(
                "No orthogonal array exists with such a parameter combination")
        return np.unique(np.array(arraylist[0]), axis=0)

    @property
    def monotonic(self):
        dx = np.diff(self.factor_levels)
        return np.all(dx <= 0) or np.all(dx >= 0)
