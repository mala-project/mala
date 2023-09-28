"""Base class for all hyperparameter optimizers."""
from abc import abstractmethod, ABC
import os

from mala.common.parallelizer import printout
from mala.common.parameters import Parameters
from mala.datahandling.data_handler import DataHandler
from mala.datahandling.data_scaler import DataScaler
from mala.network.hyperparameter import Hyperparameter
from mala.network.objective_base import ObjectiveBase


class HyperOpt(ABC):
    """Base class for hyperparameter optimizer.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __new__(cls, params: Parameters, data=None, use_pkl_checkpoints=False):
        """
        Create a HyperOpt instance.

        The correct type of hyperparameter optimizer will automatically be
        instantiated by this class if possible. You can also instantiate
        the desired hyperparameter optimizer directly by calling upon the
        subclass.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this hyperparameter optimizer.
        """
        hoptimizer = None

        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == HyperOpt:
            if params.hyperparameters.hyper_opt_method == "optuna":
                from mala.network.hyper_opt_optuna import HyperOptOptuna
                hoptimizer = super(HyperOpt, HyperOptOptuna).\
                    __new__(HyperOptOptuna)
            if params.hyperparameters.hyper_opt_method == "oat":
                from mala.network.hyper_opt_oat import HyperOptOAT
                hoptimizer = super(HyperOpt, HyperOptOAT).\
                    __new__(HyperOptOAT)
            if params.hyperparameters.hyper_opt_method == "naswot":
                from mala.network.hyper_opt_naswot import HyperOptNASWOT
                hoptimizer = super(HyperOpt, HyperOptNASWOT).\
                    __new__(HyperOptNASWOT)

            if hoptimizer is None:
                raise Exception("Unknown hyperparameter optimizer requested.")
        else:
            hoptimizer = super(HyperOpt, cls).__new__(cls)

        return hoptimizer

    def __init__(self, params: Parameters, data=None,
                 use_pkl_checkpoints=False):
        self.params: Parameters = params
        self.data_handler = data
        self.objective = ObjectiveBase(self.params, self.data_handler)
        self.use_pkl_checkpoints = use_pkl_checkpoints

    def add_hyperparameter(self, opttype="float", name="", low=0, high=0,
                           choices=None):
        """
        Add a hyperparameter to the current investigation.

        Parameters
        ----------
        opttype : string
            Datatype of the hyperparameter. Follows optuna's naming
            conventions.
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
                Hyperparameter(self.params.hyperparameters.
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

    def _save_params_and_scaler(self):
        # Saving the Scalers is straight forward.
        iscaler_name = self.params.hyperparameters.checkpoint_name \
            + "_iscaler.pkl"
        oscaler_name = self.params.hyperparameters.checkpoint_name \
            + "_oscaler.pkl"
        self.data_handler.input_data_scaler.save(iscaler_name)
        self.data_handler.output_data_scaler.save(oscaler_name)

        # For the parameters we have to make sure we choose the correct
        # format.
        if self.use_pkl_checkpoints:
            param_name = self.params.hyperparameters.checkpoint_name \
                         + "_params.pkl"
            self.params.save_as_pickle(param_name)
        else:
            param_name = self.params.hyperparameters.checkpoint_name \
                         + "_params.json"
            self.params.save_as_json(param_name)

    @classmethod
    def checkpoint_exists(cls, checkpoint_name, use_pkl_checkpoints=False):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        if use_pkl_checkpoints:
            param_name = checkpoint_name + "_params.pkl"
        else:
            param_name = checkpoint_name + "_params.json"

        return all(map(os.path.isfile, [iscaler_name, oscaler_name,
                                        param_name]))

    @classmethod
    def _resume_checkpoint(cls, checkpoint_name, no_data=False,
                           use_pkl_checkpoints=False):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which the checkpoint is loaded.

        no_data : bool
            If True, the data won't actually be loaded into RAM or scaled.
            This can be useful for cases where a checkpoint is loaded
            for analysis purposes.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOptuna
            The hyperparameter optimizer reconstructed from the checkpoint.
        """
        printout("Loading hyperparameter optimization from checkpoint.",
                 min_verbosity=0)
        # The names are based upon the checkpoint name.
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        if use_pkl_checkpoints:
            param_name = checkpoint_name + "_params.pkl"
            loaded_params = Parameters.load_from_pickle(param_name)
        else:
            param_name = checkpoint_name + "_params.json"
            loaded_params = Parameters.load_from_json(param_name)
        optimizer_name = checkpoint_name + "_hyperopt.pth"

        # First load the all the regular objects.
        loaded_iscaler = DataScaler.load_from_file(iscaler_name)
        loaded_oscaler = DataScaler.load_from_file(oscaler_name)

        printout("Preparing data used for last checkpoint.", min_verbosity=1)
        # Create a new data handler and prepare the data.
        if no_data is True:
            loaded_params.data.use_lazy_loading = True
        new_datahandler = DataHandler(loaded_params,
                                      input_data_scaler=loaded_iscaler,
                                      output_data_scaler=loaded_oscaler,
                                      clear_data=False)
        new_datahandler.prepare_data(reparametrize_scaler=False)

        return loaded_params, new_datahandler, optimizer_name
