"""Hyperparameter to use with optuna."""
from optuna.trial import Trial

from mala.network.hyperparameter import Hyperparameter


class HyperparameterOptuna(Hyperparameter):
    """Represents an optuna parameter.

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

    def __init__(self, hotype=None, opttype="float", name="", low=0, high=0, choices=None):
        super(HyperparameterOptuna, self).__init__(opttype=opttype,
                                                   name=name,
                                                   low=low,
                                                   high=high,
                                                   choices=choices)

        # For now, only three types of hyperparameters are allowed:
        # Lists, floats and ints.
        if self.opttype != "float" and self.opttype != "int" and self.opttype \
                != "categorical":
            raise Exception("Unsupported Hyperparameter type.")

    def get_parameter(self, trial: Trial):
        """
        Extract current value of hyperparameter from an optuna Trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial, from which the hyperparameter value should be
            extracted.

        Returns
        -------
        return_value : float, int or string
            Return value is based on type of hyperparameter.
        """
        if self.opttype == "float":
            return self.get_float(trial)
        if self.opttype == "int":
            return self.get_int(trial)
        if self.opttype == "categorical":
            return self.get_categorical(trial)
        raise Exception("Wrong hyperparameter type.")

    def get_float(self, trial: Trial):
        """
        Extract float hyperparameter from an optuna Trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial, from which the hyperparameter value should be
            extracted.

        Returns
        -------
        return_value : float, int or string
            Return value is based on type of hyperparameter.
        """
        if self.opttype == "float":
            return trial.suggest_float(self.name, self.low, self.high)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_int(self, trial: Trial):
        """
        Extract integer hyperparameter from an optuna Trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial, from which the hyperparameter value should be
            extracted.

        Returns
        -------
        return_value : int
            Return value is based on type of hyperparameter.
        """
        if self.opttype == "int":
            return trial.suggest_int(self.name, self.low, self.high)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_categorical(self, trial: Trial):
        """
        Extract categorical (string) hyperparameter from an optuna Trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial, from which the hyperparameter value should be
            extracted.

        Returns
        -------
        return_value : string
            Return value is based on type of hyperparameter.
        """
        if self.opttype == "categorical":
            if len(self.choices) > 1:
                return trial.suggest_categorical(self.name, self.choices)
            else:
                return self.choices[0]
        else:
            raise Exception("Wrong hyperparameter type.")
