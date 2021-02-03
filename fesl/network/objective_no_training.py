import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from fesl.common.parameters import Parameters
from fesl.datahandling.handler_base import HandlerBase
from fesl.network.network import Network
from fesl.network.objective_base import ObjectiveBase
from fesl.common.parameters import printout

import matplotlib.pyplot as plt

class ObjectiveNoTraining(ObjectiveBase):

    def __init__(self, search_parameters: Parameters, data_handler: HandlerBase, trial_type):
        super(ObjectiveNoTraining, self).__init__(search_parameters, data_handler)
        self.trial_type = trial_type

    def __call__(self, trial):
        # Parse the parameters using the base class.
        super(ObjectiveNoTraining, self).parse_trial(trial)

        # Build the network.
        net = Network(self.params)
        device = "cuda" if torch.cuda.is_available() and self.params.training.use_gpu else "cpu"
        net.to(device)

        # Load the batchesand get the jacobian.
        loader = DataLoader(self.data_handler.training_data_set,
                            batch_size=self.params.training.mini_batch_size,
                            shuffle=True)
        jac = ObjectiveNoTraining._get_batch_jacobian(net, loader, device)

        # Loss = - score!
        surrogate_loss = float('inf')
        try:
            surrogate_loss = - ObjectiveNoTraining._calc_score(jac)
        except:
            printout("Got a NaN, ignoring sample.")
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
    def corrcoef(x):
        # Pytorch implementation of np.corrcoef
        mean_x = torch.mean(x, 1, True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        c = torch.clamp(c, -1.0, 1.0)

        return c

    @staticmethod
    def _calc_score(jacobian: Tensor):
        correlations = ObjectiveNoTraining.corrcoef(jacobian)
        eigen_values, _ = torch.eig(correlations)
        # Only consider the real valued part, imaginary part is rounding artefact
        eigen_values = eigen_values.type(torch.float32)
        # Needed for numerical stability. 1e-4 instead of 1e-5 in reference as the torchs eigenvalue solver on GPU
        # seems to have bigger rounding errors than numpy, resulting in slightly larger negative Eigenvalues
        k = 1e-4
        v = -torch.sum(torch.log(eigen_values + k) + 1. / (eigen_values+k))
        return v