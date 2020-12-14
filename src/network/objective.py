from optuna import Trial
from .network import Network
from .trainer import Trainer
from .optuna_parameter import OptunaParameter

class Objective:
    """Wraps the training process."""
    def __init__(self, p, dh):
        self.params = p
        self.data_handler = dh

    def __call__(self, trial: Trial):
        # parse hyperparameter list.
        par: OptunaParameter
        for par in self.params.hyperparameters.hyperparameter_list:
            if (par.name == "learning_rate"):
                self.params.training.learning_rate = par.get_parameter(trial)

        # Perform training and report best test loss back to optuna.
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss
