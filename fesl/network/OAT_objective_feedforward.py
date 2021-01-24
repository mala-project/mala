from .network import Network
from .trainer import Trainer
from .OAT_parameter import OA_Parameter

class OA_ObjectiveFeedForward():
    def __init__(self, params,dh):
        self.params= params
        self.data_handler= dh
    def __call__(self, trial):
        par: OA_Parameter
        for factor_idx, par in enumerate(self.params.hyperparameters.hlist):
            if "layer_activation" in par.name:
                self.params.network.layer_activations.append(par.get_parameter(trial,factor_idx))
            elif "trainingtype" in par.name:
                self.params.training.trainingtype= par.get_parameter(trial, factor_idx)
            else:
                print("Optimization of hyperparameter ", par.name, "not supported at the moment.")
        
        # Perform training and report best test loss back to optuna.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss