"""Hyperparameter to use with OAT."""

from mala.network.hyperparameter import Hyperparameter


class HyperparameterOAT(Hyperparameter):
    """Represents a hyperparameter for OAT.

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

    choices :
        List of possible choices (for categorical parameter).
    """

    def __init__(self, hotype=None, opttype="categorical", name="", choices=[],
                 low=0, high=0):
        super(HyperparameterOAT, self).__init__(opttype=opttype,
                                                name=name,
                                                choices=choices)

        if self.opttype != "categorical":
            raise Exception("Unsupported Hyperparameter type.")

    @property
    def num_choices(self):
        """Get the number of choices for this hyperparameter."""
        return len(self.choices)

    @num_choices.setter
    def num_choices(self, value):
        pass

    def get_parameter(self, trial, idx):
        """
        Extract current value of hyperparameter from an orthogonal array.

        Parameters
        ----------
        trial : numpy.array
            Row in an orthogonal array which respresents current trial.

        idx : int
            Index of current hyperparameter in OA row.

        Returns
        -------
        return_value : float, int or string
            Return value is based on type of hyperparameter.
        """
        if self.opttype == "categorical":
            return self.get_categorical(trial, idx)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_categorical(self, trial, idx):
        """
        Extract categorical (string) hyperparameter from an optuna Trial.

        Parameters
        ----------
        trial : numpy.array
            Row in an orthogonal array which respresents current trial.

        idx : int
            Index of current hyperparameter in OA row.

        Returns
        -------
        return_value : string
            Return value is based on type of hyperparameter.
        """
        return self.choices[trial[idx]]
