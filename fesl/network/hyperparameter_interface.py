from .hyperparameter_oat import HyperparameterOAT
from .hyperparameter_optuna import HyperparameterOptuna


def HyperparameterInterface(hotype, opttype="float", name="", low=0, high=0, choices=[]):
    """Defines an interface which, based on the choices made by the user
    in parameters, will always return the right kind of hyperparameter optimizer."""

    hparam = None
    if hotype == "optuna":
        hparam = HyperparameterOptuna(opttype=opttype, name=name, low=low, high=high, choices=choices)
    elif hotype == "oat":
        hparam = HyperparameterOAT(opttype=opttype, name=name, choices=choices)

    if hparam is not None:
        return hparam
    else:
        raise Exception("Unknown hyperparameter type..")
