"""Hyperparameter optimizer using orthogonal array tuning."""
import oapackage as oa
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase
import numpy as np
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
        self.best_trial = None
        self.n_factors = None
        self.n_levels = None
        self.strength = None
        self.N_runs = None
        self.objective = ObjectiveBase(self.params, self.data_handler)

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        In this case, these are choosen based on an orthogonal array.
        """
        number_of_trial = 0
        # The parameters could have changed.
        self.objective = ObjectiveBase(self.params, self.data_handler)
        for row in self.orthogonal_arr:
            printout("Trial number", number_of_trial)
            self.trial_losses.append(self.objective(row))
            number_of_trial += 1

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
        self.best_trial = self.orthogonal_arr[idx]
        self.objective.parse_trial_oat(self.best_trial)

    def set_parameters(self, trial):
        """
        Set the parameters to a specific trial.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        self.objective.parse_trial_oat(trial)

    @property
    def orthogonal_arr(self):
        """Orthogonal array used for optimal hyperparameter sampling."""
        arrayclass = oa.arraydata_t(self.n_levels, self.N_runs, self.strength,
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
        self.n_factors = len(self.params.hyperparameters.hlist)

        # if self.n_factors>4:
        #     raise Exception("Sorry only upto 3 factors are supported")

        self.n_levels = min([par.num_choices for par in self.params.
                            hyperparameters.hlist])
        self.strength = 3
        self.N_runs = pow(self.n_levels, self.n_factors)
