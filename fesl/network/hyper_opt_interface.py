"""Interface to get correct hyperparameter optimizer."""
from .hyper_opt_notraining import HyperOptNoTraining
from .hyper_opt_oat import HyperOptOAT
from .hyper_opt_optuna import HyperOptOptuna


def HyperOptInterface(params):
    """
    Get the correct hyperparameter optimizer based on the parameters provided.

    Parameters
    ----------
        params : fesl.common.parametes.Parameters
            Parameters used to create a hyperparameter optimizer.

    Returns
    -------
    hyperparameter_optimizer : fesl.network.hyper_opt_base.HyperOptBase
        The desired hyperparameter optimizer. 
    """
    hoptimizer = None
    if params.hyperparameters.hyper_opt_method == "optuna":
        hoptimizer = HyperOptOptuna(params)
    if params.hyperparameters.hyper_opt_method == "oat":
        hoptimizer = HyperOptOAT(params)
    if params.hyperparameters.hyper_opt_method == "notraining":
        hoptimizer = HyperOptNoTraining(params)

    if hoptimizer is not None:
        return hoptimizer
    else:
        raise Exception("Unknown hyperparameter optimizer requested.")
