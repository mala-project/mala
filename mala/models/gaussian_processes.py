import gpytorch
import torch

from mala import DataHandler
from mala.common.parameters import ParametersModel


class GaussianProcesses(gpytorch.models.ExactGP):
    def __init__(self, params, data_handler: DataHandler):
        self.params: ParametersModel = params.model

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # Parse parameters.
        # Likelihood.
        likelihood = None
        if self.params.loss_function_type == "gaussian_likelihood":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if likelihood is None:
            raise Exception("Invalid Likelihood selected.")

        super(GaussianProcesses, self).__init__(data_handler.
                                                training_data_inputs,
                                                data_handler.
                                                training_data_outputs,
                                                likelihood)
        # Mean.
        self.mean_module = None
        if self.params.gp_mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()

        if self.mean_module is None:
            raise Exception("Invalid mean module selected.")

        # Kernel.
        self.covar_module = None
        if self.params.kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if self.params.kernel == "linear":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

        if self.covar_module is None:
            raise Exception("Invalid kernel selectded.")

        # Multivariate distribution.
        self.multivariate_distribution = None
        if self.params.multivariate_distribution == "normal":
            self.multivariate_distribution = gpytorch.distributions.MultivariateNormal

        if self.multivariate_distribution is None:
            raise Exception("Invalid multivariate distribution selected.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.multivariate_distribution(mean_x, covar_x)

    def calculate_loss(self, output, target):
        return -gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)(output, target).sum()

    # TODO: implement this.
    def load_from_file(cls, params, path_to_file):
        pass

    # TODO: Implement this.
    def save_model(self, path_to_file):
        pass

    def train(self, mode=True):
        self.likelihood.train()
        return super(GaussianProcesses, self).train(mode=mode)

    def eval(self):
        self.likelihood.eval()
        return super(GaussianProcesses, self).eval()

