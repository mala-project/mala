from abc import abstractmethod, ABC

from .hyperparameter_interface import HyperparameterInterface


class HyperOptBase(ABC):
    """Base class for hyperparameter optimization."""

    def __init__(self, params):
        self.params = params

    def add_hyperparameter(self, opttype="float", name="", low=0, high=0, choices=[]):
        self.params.hyperparameters.hlist.append(HyperparameterInterface(self.params.hyperparameters.hyper_opt_method,
                                                                         opttype=opttype, name=name, low=low,
                                                                         high=high, choices=choices))

    def clear_hyperparameters(self):
        self.params.hyperparameters.hlist = []

    @abstractmethod
    def perform_study(self, datahandler):
        """
        Performs the study, i.e. the optimization by sampling a certain subset of network architectures.

        Parameters
        ----------
        datahandler : fesl.datahandling.HandlerBase or derivative thereof
            datahandler to be used during the hyperparameter optimization.

        Returns
        -------
        """
        pass

    @abstractmethod
    def set_optimal_parameters(self):
        """
        Sets the optimal parameters found in the present study.

        Parameters
        ----------
        Returns
        -------
        """
        pass