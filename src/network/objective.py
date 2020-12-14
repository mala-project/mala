from optuna import Trial
from .network import Network
from .trainer import Trainer


class Objective:
    """Wraps the training process."""
    def __init__(self, p, dh):
        self.params = p
        self.data_handler = dh

    def __call__(self, trial: Trial):

        self.params.training.learning_rate = trial.suggest_float('learning_rate', 0.0000001, 0.001)
        test_network = Network(self.params)
        test_trainer = Trainer(self.params)
        test_trainer.train_network(test_network, self.data_handler)
        return test_trainer.final_test_loss

        return self.params.training.learning_rate