from .hyper_opt_notraining import HyperOptNoTraining
from .hyper_opt_oat import HyperOptOAT
from .hyper_opt_optuna import HyperOptOptuna


def HyperOptInterface(params):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of hyperparameter optimizer."""

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
