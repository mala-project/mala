"""Interface function to get the correct type of hyperparameter."""
from mala.common.json_serializable import JSONSerializable


class Hyperparameter(JSONSerializable):
    """
    Base class for all hyperparameters.

    Will always return the correct subclass of hyperparameter.

    Parameters
    ----------
    hotype : string
        Type of hyperparameter. Currently supported:

            - optuna: Hyperparameter for HyperOptOptuna.
            - oat: Hyperparameter for HyperOptOAT.
            - naswot: Hyperparameter for HyperOptNASWOT.

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

    choices : list
        List of possible choices (for categorical parameter).

    Returns
    -------
    hyperparameter : HyperparameterOptuna or HyperparameterOAT or HyperparameterNASWOT or HyperparameterACSD
        Hyperparameter in desired format.
    """

    def __new__(cls, hotype=None, opttype="float", name="", low=0, high=0,
                choices=None):
        """
        Create a Hyperparameter instance.

        The correct type of hyperparamer calculator will automatically be
        instantiated by this class if possible. You can also instantiate
        the desired hyperparamer directly by calling upon the subclass.

        Parameters
        ----------
        hotype : string
            Type of hyperparameter. Currently supported:

                - optuna: Hyperparameter for HyperOptOptuna.
                - oat: Hyperparameter for HyperOptOAT.
                - naswot: Hyperparameter for HyperOptNASWOT.

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

        choices : list
            List of possible choices (for categorical parameter).
        """
        hparam = None
        if cls == Hyperparameter:
            if hotype == "optuna":
                from mala.network.hyperparameter_optuna import \
                    HyperparameterOptuna
                hparam = HyperparameterOptuna(hotype=hotype,
                                              opttype=opttype, name=name,
                                              low=low,
                                              high=high, choices=choices)
            if hotype == "naswot":
                from mala.network.hyperparameter_naswot import \
                    HyperparameterNASWOT
                hparam = HyperparameterNASWOT(hotype=hotype,
                                              opttype=opttype, name=name,
                                              low=low,
                                              high=high, choices=choices)
            if hotype == "oat":
                from mala.network.hyperparameter_oat import HyperparameterOAT
                hparam = HyperparameterOAT(hotype=hotype,
                                           opttype=opttype, name=name,
                                           choices=choices)
            if hotype == "acsd":
                from mala.network.hyperparameter_acsd import HyperparameterACSD
                hparam = HyperparameterACSD(hotype=hotype,
                                            opttype=opttype, name=name,
                                            low=low, high=high, choices=choices)

            if hparam is None:
                raise Exception("Unsupported hyperparameter.")
        else:
            hparam = super(Hyperparameter, cls).__new__(cls)
        return hparam

    def __init__(self, hotype=None, opttype="float", name="", low=0, high=0,
                 choices=None):
        super(Hyperparameter, self).__init__()
        self.opttype = opttype
        self.name = name
        self.low = low
        self.high = high
        self.choices = choices
