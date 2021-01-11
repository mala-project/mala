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


class NoTrainingObjective(ObjectiveBase):

    def __init__(self, search_parameters: Parameters, data_handler: HandlerBase):
        super(NoTrainingObjective, self).__init__(search_parameters, data_handler)

    def __call__(self, trial: Trial):
        net = Network(self.params)
        device = "cuda" if torch.cuda.is_available() and self.params.use_gpu else "cpu"

        net.to(device)

        loader = DataLoader(self.data_handler.training_data_set,
                            batch_size=self.params.mini_batch_size,
                            shuffle=True)
        jac = NoTrainingObjective._get_batch_jacobian(net, loader, device)
        jacobian = jac.cpu().numpy()
        correlations = np.corrcoef(jacobian)
        plt.hist(correlations.flatten(), bins=100)
        plt.show()

        return NoTrainingObjective._calc_score(jacobian)


    @staticmethod
    def _get_batch_jacobian(net: Network, loader: DataLoader, device) -> Tensor:
        x: Tensor
        (x, _) = next(iter(loader))
        x = x.to(device)
        net.zero_grad()
        y : Tensor= net(x)
        y.backward(torch.ones_like(y))
        jacobian = x.grad.detach()
        return jacobian

    @staticmethod
    def _calc_score(jacobian: np.ndarray):
        correlations = np.corrcoef(jacobian)
        eigen_values, _ = np.linalg.eig(correlations)
        k = 1e-5
        return -np.sum(np.log(eigen_values + k) + 1. / (eigen_values+k))