"""Base class for all hyperparameter optimizers."""
from abc import abstractmethod, ABC
from .hyperparameter_interface import HyperparameterInterface
from .objective_base import ObjectiveBase


class HyperOptBase(ABC):
    """Base class for hyperparameter optimizater.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.
    """

    def __init__(self, params, data):
        self.params = params
        self.data_handler = data
        self.objective = ObjectiveBase(self.params, self.data_handler)

    def add_hyperparameter(self, opttype="float", name="", low=0, high=0,
                           choices=None):
        """
        Add a hyperparameter to the current investigation.

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
            layer sizes use e.g. ff_neurons_layer_001,
            ff_neurons_layer_002, etc. as names.

        low : float or int
            Lower bound for numerical parameter.

        high : float or int
            Higher bound for numerical parameter.

        choices :
            List of possible choices (for categorical parameter).
        """

        self.params.\
            hyperparameters.hlist.append(
                HyperparameterInterface(self.params.hyperparameters.
                                        hyper_opt_method,
                                        opttype=opttype,
                                        name=name,
                                        low=low,
                                        high=high,
                                        choices=choices))

    def clear_hyperparameters(self):
        """Clear the list of hyperparameters that are to be investigated."""
        self.params.hyperparameters.hlist = []

    @abstractmethod
    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        """
        pass

    @abstractmethod
    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        pass

    def set_parameters(self, trial):
        """
        Set the parameters to a specific trial.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        self.objective.parse_trial(trial)
