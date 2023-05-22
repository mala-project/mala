"""Objective functions for hyperparameter optimizations without training."""
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mala.common.parameters import Parameters
from mala.datahandling.data_handler import DataHandler
from mala.network.network import Network
from mala.network.objective_base import ObjectiveBase


class ObjectiveNASWOT(ObjectiveBase):
    """
    Represents the objective function using no NN training.

    The objective value is calculated using the Jacobian.

    Parameters
    ----------
    search_parameters : mala.common.parameters.Parameters
        Parameters used to create this objective.

    data_handler : mala.datahandling.data_handler.DataHandler
        datahandler to be used during the hyperparameter optimization.

    trial_type : string
        Format of hyperparameters used in this objective. Supported
        choices are:

            - optuna
            - oat

    batch_size : int
        Batch size for the forwarding. Default (None) means that the regular
        mini batch size from ParametersRunning is used; however, for some
        applications it might make sense to specify something different.
    """

    def __init__(self, search_parameters: Parameters, data_handler:
                 DataHandler, trial_type, batch_size=None):
        super(ObjectiveNASWOT, self).__init__(search_parameters,
                                              data_handler)
        self.trial_type = trial_type
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = search_parameters.running.mini_batch_size

    def __call__(self, trial):
        """
        Get objective value for a trial (=set of hyperparameters).

        Parameters
        ----------
        trial
            A trial is a set of hyperparameters; can be an optuna based
            trial or simply a OAT compatible list.
        """
        # Parse the parameters using the base class.
        super(ObjectiveNASWOT, self).parse_trial(trial)

        # Build the network.
        surrogate_losses = []
        for i in range(0, self.params.hyperparameters.
                       number_training_per_trial):
            net = Network(self.params)
            device = self.params.device

            # Load the batchesand get the jacobian.
            do_shuffle = self.params.running.use_shuffling_for_samplers
            if self.data_handler.parameters.use_lazy_loading or \
                    self.params.use_horovod:
                do_shuffle = False
            if self.params.running.use_shuffling_for_samplers:
                self.data_handler.mix_datasets()
            loader = DataLoader(self.data_handler.training_data_sets[0],
                                batch_size=self.batch_size,
                                shuffle=do_shuffle)
            jac = ObjectiveNASWOT.__get_batch_jacobian(net, loader, device)

            # Loss = - score!
            surrogate_loss = float('inf')
            try:
                surrogate_loss = - ObjectiveNASWOT.__calc_score(jac)
                surrogate_loss = surrogate_loss.cpu().detach().numpy().astype(
                    np.float64)
            except RuntimeError:
                print("Got a NaN, ignoring sample.")
            surrogate_losses.append(surrogate_loss)

        if self.params.hyperparameters.number_training_per_trial > 1:
            print("Losses from multiple runs are: ")
            print(surrogate_losses)

        if self.params.hyperparameters.trial_ensemble_evaluation == "mean":
            return np.mean(surrogate_losses)

        elif self.params.hyperparameters.trial_ensemble_evaluation == \
                "mean_std":
            mean = np.mean(surrogate_losses)

            # Cannot calculate the standar deviation of a bunch of infinities.
            if np.isinf(mean):
                return mean
            else:
                return np.mean(surrogate_losses) + \
                       np.std(surrogate_losses)
        else:
            raise Exception("No way to estimate the trial metric from ensemble"
                            " training provided.")

    @staticmethod
    def __get_batch_jacobian(net: Network, loader: DataLoader, device) \
            -> Tensor:
        """Calculate the jacobian of the batch."""
        x: Tensor
        (x, _) = next(iter(loader))
        x = x.to(device)
        net.zero_grad()
        x.requires_grad_(True)
        y: Tensor = net(x)
        y.backward(torch.ones_like(y))
        jacobian = x.grad.detach()
        return jacobian

    @staticmethod
    def __corrcoef(x):
        """
        Calculate the correlation coefficient of an array.

        Pytorch implementation of np.corrcoef.
        """
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
    def __calc_score(jacobian: Tensor):
        """Calculate the score for a Jacobian."""
        correlations = ObjectiveNASWOT.__corrcoef(jacobian)
        eigen_values, _ = torch.linalg.eig(correlations)
        # Only consider the real valued part, imaginary part is rounding
        # artefact
        eigen_values = eigen_values.real
        # Needed for numerical stability. 1e-4 instead of 1e-5 in reference
        # as the torchs eigenvalue solver on GPU
        # seems to have bigger rounding errors than numpy, resulting in
        # slightly larger negative Eigenvalues
        k = 1e-4
        v = -torch.sum(torch.log(eigen_values + k) + 1. / (eigen_values+k))
        return v
