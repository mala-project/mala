import oapackage as oa
from .hyper_opt_base import HyperOptBase
from .objective_base import ObjectiveBase
import numpy as np


class HyperOptOAT(HyperOptBase):
    def __init__(self, params):
        super(HyperOptOAT,self).__init__(params)
        self.objective = None
        self.trial_losses = []
        self.best_trial = None
        self.n_factors = None
        self.n_levels = None
        self.strength = None
        self.N_runs = None

    def perform_study(self, data_handler):
        """Runs the OAT "study" """
        number_of_trial = 0
        self.objective = ObjectiveBase(self.params, data_handler)
        for row in self.orthogonal_arr:
            print("Trial number", number_of_trial)
            self.trial_losses.append(self.objective(row))
            number_of_trial += 1

        # Return the best lost value we could achieve.
        return min(self.trial_losses)

    def set_optimal_parameters(self):
        # Getting the best trial based on the test errors
        idx = self.trial_losses.index(min(self.trial_losses))
        self.best_trial = self.orthogonal_arr[idx]
        self.objective.parse_trial_oat(self.best_trial)

    @property
    def orthogonal_arr(self):
        arrayclass = oa.arraydata_t(self.n_levels, self.N_runs, self.strength, self.n_factors)
        arraylist = [arrayclass.create_root()]

        # extending the orthogonal array
        options = oa.OAextend()
        options.setAlgorithmAuto(arrayclass)

        for _ in range(self.strength + 1, self.n_factors + 1):
            arraylist_extensions = oa.extend_arraylist(arraylist, arrayclass, options)
            dd = np.array([a.Defficiency() for a in arraylist_extensions])
            idxs = np.argsort(dd)
            arraylist = [arraylist_extensions[ii] for ii in idxs]

        return np.unique(np.array(arraylist[0]), axis=0)


    def add_hyperparameter(self, opttype="float", name="", low=0, high=0, choices=[]):
        super(HyperOptOAT, self).add_hyperparameter(opttype=opttype, name=name, low=low, high=high, choices=choices)
        self.n_factors = len(self.params.hyperparameters.hlist)

        # if self.n_factors>4:
        #     raise Exception("Sorry only upto 3 factors are supported")

        self.n_levels = min([par.num_choices for par in self.params.hyperparameters.hlist])
        self.strength = 3
        self.N_runs = pow(self.n_levels, self.n_factors)

