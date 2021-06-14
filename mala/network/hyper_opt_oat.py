"""Hyperparameter optimizer using orthogonal array tuning."""
import oapackage as oa
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase
import numpy as np
import itertools
from mala.common.parameters import printout


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
        self.optimal_params = None
        self.n_factors = None
        self.factor_levels = None
        self.strength = None
        self.N_runs = None
        self.OA = None

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, these are choosen based on an orthogonal array.
        """
        self.n_factors = len(self.params.hyperparameters.hlist)

        self.factor_levels = [par.num_choices for par in self.params.
                              hyperparameters.hlist]
        self.strength = 3
        self.N_runs = self.number_of_runs()
        self.OA = self.get_orthogonal_array()
        number_of_trial = 0
        self.objective = ObjectiveBase(self.params, self.data_handler)
        for row in self.OA:
            printout("Trial number", number_of_trial)
            self.trial_losses.append(self.objective(row))
            number_of_trial += 1

        # Return the best lost value we could achieve.
        return min(self.trial_losses)

    def set_optimal_parameters(self):
        """
        Find the optimal set of hyperparameters by doing range analysis.
        This is done using loss instead of accuracy as done in the paper.

        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """

        def indices(idx, val): return np.where(self.OA[:, idx] == val)[0]

        R = [[self.trial_losses[indices(i, l)] for l in range(levels)]
             for (i, levels) in enumerate(self.factor_levels)]

        A = [[i/len(j) for i in j] for j in R]

        self.optimal_params = np.array([i.index(max(i)) for i in A])
        importance = np.argsort([max(i)-min(i) for i in A])

        print("Order of Importance: ")
        printout(
            [self.params.hyperparameters.hlist[idx].name for idx in self.optimal_params], " > ")

        print("Optimal Hyperparameters:")
        self.objective.parse_trial_oat(self.optimal_params)

    def get_orthogonal_array(self):
        """Generate the best Orthogonal array used for optimal hyperparameter sampling."""

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

        return np.unique(np.array(arraylist[0]), axis=0)

    def add_hyperparameter(self, opttype="float", name="", low=0, high=0,
                           choices=None):
        """
        Add a hyperparameter to the current investigation.

        Parameters
        ----------
        opttype : string
            Datatype of the hyperparameter. Follows optunas naming convetions.
            Currently supported are:

                - categorical (list)

        name : string
            Name of the hyperparameter. Please note that these names always
            have to be distinct; if you e.g. want to investigate multiple
            layer sizes use e.g. ff_neurons_layer_001, ff_neurons_layer_002,
            etc. as names.

        low : float or int
            Currently unsupported: Lower bound for numerical parameter.

        high : float or int
            Currently unsupported: Higher bound for numerical parameter.

        choices :
            List of possible choices (for categorical parameter).
        """
        super(HyperOptOAT, self).add_hyperparameter(opttype=opttype, name=name,
                                                    low=low, high=high,
                                                    choices=choices)

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
        return N
