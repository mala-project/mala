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
        self.strength= 2
        self.N_runs= pow(self.n_levels,self.n_factors)
        self.objective= None 
        self.trial_losses= None

    def perform_study(self, data_handler):
        self.objective= OA_ObjectiveFeedForward(self.params, data_handler)
        self.trial_losses=map(self.objective, self.create_oat)

    @property
    def create_oat(self):
        arrayclass= oa.arraydata_t(self.n_levels, self.N_runs, self.strength, self.n_factors)
        arraylist= [arrayclass.create_root()]

        for _ in range(self.strength+1, self.n_factors+1):
            arraylist= oa.extend_arraylist(arraylist, arrayclass)

        return np.array(arraylist[0])

    def set_optimal_parameters(self):
        pass
