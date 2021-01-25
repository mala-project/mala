import numpy as np
import torch
from optuna import Trial
from torch import Tensor
from torch.utils.data import DataLoader

from fesl.common.parameters import Parameters
from fesl.datahandling.handler_base import HandlerBase
from fesl.network.network import Network
from fesl.network.objective_base import ObjectiveBase

import matplotlib.pyplot as plt

from fesl.network.trainer import Trainer


class NoTrainingObjective(ObjectiveBase):

    def __init__(self, search_parameters: Parameters, data_handler: HandlerBase):
        super(NoTrainingObjective, self).__init__(search_parameters, data_handler)
        self.samples = []

    def __call__(self, trial: Trial):
        super(NoTrainingObjective, self).__call__(trial)
        net = Network(self.params)
        device = "cuda" if torch.cuda.is_available() and self.params.training.use_gpu else "cpu"

        net.to(device)

        loader = DataLoader(self.data_handler.training_data_set,
                            batch_size=self.params.training.mini_batch_size,
                            shuffle=True)
        jac = NoTrainingObjective._get_batch_jacobian(net, loader, device)

        # also train to see how good this metric actually is
        trainer = Trainer(self.params)
        trainer.train_network(net, self.data_handler)
        actual_loss = trainer.final_test_loss

        # Loss = - score!
        surrogate_loss = - NoTrainingObjective._calc_score(jac.cpu().numpy())

        self.samples.append((actual_loss, surrogate_loss))
        return surrogate_loss


    @staticmethod
    def plot_correlations(jacobian: torch.Tensor):
        jacobian = jacobian.cpu().numpy()
        correlations = np.corrcoef(jacobian)
        plt.hist(correlations.flatten(), bins=100)
        plt.show()


    @staticmethod
    def _get_batch_jacobian(net: Network, loader: DataLoader, device) -> Tensor:
        x: Tensor
        (x, _) = next(iter(loader))
        x = x.to(device)
        net.zero_grad()
        x.requires_grad_(True)
        y : Tensor= net(x)
        y.backward(torch.ones_like(y))
        jacobian = x.grad.detach()
        return jacobian

    @staticmethod
    def _calc_score(jacobian: np.array):
        # No pytorch implementation for corrcoeff so use numpy
        correlations = np.corrcoef(jacobian)
        eigen_values, _ = np.linalg.eig(correlations)
        eigen_values = eigen_values.astype(np.float32)
        k = 1e-5
        return -np.sum(np.log(eigen_values + k) + 1. / (eigen_values+k))

    def set_optimal_parameters(self, study):
        # abuse this for getting some insight
        a = np.array(self.samples).T
        fig, axs = plt.subplots()
        axs.scatter(x = a[0,:], y = a[1, :])
        axs.set_xlabel("training loss")
        axs.set_ylabel("surrogate loss")
        axs.set_title("surrogate vs training loss")
        fig.savefig("surrogate_loss_correlation.png")