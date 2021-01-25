from .network import Network
from .trainer import Trainer
from .OAT_parameter import OA_Parameter

class OA_ObjectiveFeedForward():
    def __init__(self, params,dh):
        self.params= params
        self.data_handler= dh

        # We need to find out if we have to reparametrize the lists with the layers and the activations.
        self.optimize_activation_list = False
        par: OA_Parameter
        for par in self.params.hyperparameters.hlist:
            if "layer_activation" in par.name:
                self.optimize_activation_list = True

    def __call__(self, trial):

        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: OA_Parameter
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial,factor_idx))
            elif "trainingtype" in par.name:
                self.params.training.trainingtype= par.get_parameter(trial, factor_idx)
            else:
                print("Optimization of hyperparameter ", par.name, "not supported at the moment.")
        
        # Perform training and report best test loss.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss
    
    def set_optimal_parameters(self, trial):
        if self.optimize_activation_list:
            self.params.network.layer_activations = []

        par: OA_Parameter
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial,factor_idx))
            elif "trainingtype" in par.name:
                self.params.training.trainingtype= par.get_parameter(trial, factor_idx)