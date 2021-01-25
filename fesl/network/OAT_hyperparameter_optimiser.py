import oapackage as oa 
from ..common.parameters import Parameters
from .OAT_objective_feedforward import OA_ObjectiveFeedForward
import numpy as np

class OAT_HyperOpt:
    def __init__(self, params):
        params : Parameters
        self.params= params
        self.n_factors=len(params.hyperparameters.hlist)

        # if self.n_factors>4: 
        #     raise Exception("Sorry only upto 3 factors are supported")

        self.n_levels= min([par.num_choices for par in params.hyperparameters.hlist])
        self.strength= 3
        self.N_runs= pow(self.n_levels,self.n_factors)
        self.objective= None 
        self.trial_losses= None
        self.best_trial= None

    def perform_study(self, data_handler):
        self.objective= OA_ObjectiveFeedForward(self.params, data_handler)
        print(self.orthogonal_arr)
        self.trial_losses=  [self.objective(row) for row in self.orthogonal_arr]

    @property
    def orthogonal_arr(self):
        arrayclass= oa.arraydata_t(self.n_levels, self.N_runs, self.strength, self.n_factors)
        arraylist= [arrayclass.create_root()]

        for _ in range(self.strength+1, self.n_factors+1):
            arraylist= oa.extend_arraylist(arraylist, arrayclass)
        return np.unique(np.array(arraylist[0]), axis=0)

    def set_optimal_parameters(self):

        #Getting the best trial based on the test errors
        idx= self.trial_losses.index(min(self.trial_losses))
        self.best_trial= self.orthogonal_arr[idx]
        print("The best parameters",self.best_trial)
        print("Final test loss:",self.trial_losses[idx])
        self.objective.set_optimal_parameters(self.best_trial)
