"""Hyperparameter to use with optuna."""
from mala.network.hyperparameter_optuna import HyperparameterOptuna


class HyperparameterNASWOT(HyperparameterOptuna):
    """Represents a parameter used for pure NASWOT.

    Based upon the hyperparameter representation for optuna.

    Parameters
    ----------
    opttype : string
        Datatype of the hyperparameter. Follows optunas naming convetions.
        In principle supported are:

            - float
            - int
            - categorical (list)

        Float and int are not available for OA based approaches at the
        moment.

    name : string
        Name of the hyperparameter. Please note that these names always
        have to be distinct; if you e.g. want to investigate multiple
        layer sizes use e.g. ff_neurons_layer_001, ff_neurons_layer_002,
        etc. as names.

    low : float or int
        Lower bound for numerical parameter.

    high : float or int
        Higher bound for numerical parameter.

    choices :
        List of possible choices (for categorical parameter).
    """

    def __init__(self, hotype=None, opttype="categorical", name="", low=0, high=0,
                 choices=None):
        super(HyperparameterNASWOT, self).__init__(opttype=opttype,
                                                   name=name,
                                                   low=low,
                                                   high=high,
                                                   choices=choices)

        # For NASWOT, only categoricals are allowed.
        if self.opttype != "categorical":
            raise Exception("Unsupported Hyperparameter type.")
