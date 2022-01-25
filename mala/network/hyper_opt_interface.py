"""Interface to get correct hyperparameter optimizer."""
from mala.network.hyper_opt_naswot import HyperOptNASWOT
from mala.network.hyper_opt_oat import HyperOptOAT
from mala.network.hyper_opt_optuna import HyperOptOptuna


def HyperOptInterface(params, data):
    """
    Get the correct hyperparameter optimizer based on the parameters provided.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create a hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.

    Returns
    -------
    hyperparameter_optimizer : mala.network.hyper_opt_base.HyperOptBase
        The desired hyperparameter optimizer. 
    """
    hoptimizer = None
    if params.hyperparameters.hyper_opt_method == "optuna":
        hoptimizer = HyperOptOptuna(params, data)
    if params.hyperparameters.hyper_opt_method == "oat":
        hoptimizer = HyperOptOAT(params, data)
    if params.hyperparameters.hyper_opt_method == "naswot":
        hoptimizer = HyperOptNASWOT(params, data)

    if hoptimizer is not None:
        return hoptimizer
    else:
        raise Exception("Unknown hyperparameter optimizer requested.")
