"""Collection of all parameter related classes and functions."""
import importlib
import inspect
import json
import os
import pickle
from time import sleep

horovod_available = False
try:
    import horovod.torch as hvd
    horovod_available = True
except ModuleNotFoundError:
    pass
import numpy as np
import torch

from mala.common.parallelizer import printout, set_horovod_status, \
    set_mpi_status, get_rank, get_local_rank, set_current_verbosity, \
    parallel_warn
from mala.common.json_serializable import JSONSerializable

DEFAULT_NP_DATA_DTYPE = np.float32


class ParametersBase(JSONSerializable):
    """Base parameter class for MALA."""

    def __init__(self,):
        super(ParametersBase, self).__init__()
        self._configuration = {"gpu": False, "horovod": False, "mpi": False,
                               "device": "cpu", "openpmd_configuration": {},
                               "openpmd_granularity": 1}
        pass

    def show(self, indent=""):
        """
        Print name and values of all attributes of this object.

        Parameters
        ----------
        indent : string
            The indent used in the list with which the parameter
            shows itself.

        """
        for v in vars(self):
            if v != "_configuration":
                if v[0] == "_":
                    printout(indent + '%-15s: %s' % (v[1:], getattr(self, v)),
                             min_verbosity=0)
                else:
                    printout(indent + '%-15s: %s' % (v, getattr(self, v)),
                             min_verbosity=0)

    def _update_gpu(self, new_gpu):
        self._configuration["gpu"] = new_gpu

    def _update_horovod(self, new_horovod):
        self._configuration["horovod"] = new_horovod

    def _update_mpi(self, new_mpi):
        self._configuration["mpi"] = new_mpi

    def _update_device(self, new_device):
        self._configuration["device"] = new_device

    def _update_openpmd_configuration(self, new_openpmd):
        self._configuration["openpmd_configuration"] = new_openpmd

    def _update_openpmd_granularity(self, new_granularity):
        self._configuration["openpmd_granularity"] = new_granularity

    @staticmethod
    def _member_to_json(member):
        if isinstance(member, (int, float, type(None), str)):
            return member
        else:
            return member.to_json()

    def to_json(self):
        """
        Convert this object to a dictionary that can be saved in a JSON file.

        Returns
        -------
        json_dict : dict
            The object as dictionary for export to JSON.

        """
        json_dict = {}
        members = inspect.getmembers(self,
                                     lambda a: not (inspect.isroutine(a)))
        for member in members:
            # Filter out all private members, builtins, etc.
            if member[0][0] != "_":

                # If we deal with a list or a dict,
                # we have to sanitize treat all members of that list
                # or dict separately.
                if isinstance(member[1], list):
                    if len(member[1]) > 0:
                        _member = []
                        for m in member[1]:
                            _member.append(self._member_to_json(m))
                        json_dict[member[0]] = _member
                    else:
                        json_dict[member[0]] = member[1]

                elif isinstance(member[1], dict):
                    if len(member[1]) > 0:
                        _member = {}
                        for m in member[1].keys():
                            _member[m] = self._member_to_json(member[1][m])
                        json_dict[member[0]] = _member
                    else:
                        json_dict[member[0]] = member[1]

                else:
                    json_dict[member[0]] = self._member_to_json(member[1])
        json_dict["_parameters_type"] = type(self).__name__
        return json_dict

    @staticmethod
    def _json_to_member(json_value):
        if isinstance(json_value, (int, float, type(None), str)):
            return json_value
        else:
            if isinstance(json_value, dict) and "object" in json_value.keys():
                # We have found ourselves an object!
                # We create it and give it the JSON dict, hoping it can handle
                # it. If not, then the implementation of that class has to
                # be adjusted.
                module = importlib.import_module("mala")
                class_ = getattr(module, json_value["object"])
                new_object = class_.from_json(json_value["data"])
                return new_object
            else:
                # If it is not an elementary builtin type AND not an object
                # dictionary, something is definitely off.
                raise Exception("Could not decode JSON file, error in",
                                json_value)

    @classmethod
    def from_json(cls, json_dict):
        """
        Read this object from a dictionary saved in a JSON file.

        Parameters
        ----------
        json_dict : dict
            A dictionary containing all attributes, properties, etc. as saved
            in the json file.

        Returns
        -------
        deserialized_object : JSONSerializable
            The object as read from the JSON file.

        """
        deserialized_object = cls()
        for key in json_dict:
            # Filter out all private members, builtins, etc.
            if key != "_parameters_type":

                # If we deal with a list or a dict,
                # we have to sanitize treat all members of that list
                # or dict separately.
                if isinstance(json_dict[key], list):
                    if len(json_dict[key]) > 0:
                        _member = []
                        for m in json_dict[key]:
                            _member.append(deserialized_object.
                                           _json_to_member(m))
                        setattr(deserialized_object, key, _member)
                    else:
                        setattr(deserialized_object, key, json_dict[key])

                elif isinstance(json_dict[key], dict):
                    if len(json_dict[key]) > 0:
                        _member = {}
                        for m in json_dict[key].keys():
                            _member[m] = deserialized_object.\
                                _json_to_member(json_dict[key][m])
                        setattr(deserialized_object, key, _member)

                    else:
                        setattr(deserialized_object, key, json_dict[key])

                else:
                    setattr(deserialized_object, key, deserialized_object.
                            _json_to_member(json_dict[key]))
        return deserialized_object


class ParametersNetwork(ParametersBase):
    """
    Parameters necessary for constructing a neural network.

    Attributes
    ----------
    nn_type : string
        Type of the neural network that will be used. Currently supported are
            - "feed_forward" (default)
            - "transformer"
            - "lstm"
            - "gru"


    layer_sizes : list
        A list of integers detailing the sizes of the layer of the neural
        network. Please note that the input layer is included therein.
        Default: [10,10,0]

    layer_activations: list
        A list of strings detailing the activation functions to be used
        by the neural network. If the dimension of layer_activations is
        smaller than the dimension of layer_sizes-1, than the first entry
        is used for all layers.
        Currently supported activation functions are:

            - Sigmoid (default)
            - ReLU
            - LeakyReLU

    loss_function_type: string
        Loss function for the neural network
        Currently supported loss functions include:

            - mse (Mean squared error; default)
    no_hidden_state : bool
        If True hidden and cell state is assigned to zeros for LSTM Network.
        false will keep the hidden state active
        Default: False

    bidirection: bool
        Sets lstm network size based on bidirectional or just one direction
        Default: False

    num_hidden_layers: int
        Number of hidden layers to be used in lstm or gru or transformer nets
        Default: None

    dropout: float
        Dropout rate for transformer net
        0.0 ≤ dropout ≤ 1.0
        Default: 0.0

    num_heads: int
        Number of heads to be used in Multi head attention network
        This should be a divisor of input dimension
        Default: None
    """

    def __init__(self):
        super(ParametersNetwork, self).__init__()
        self.nn_type = "feed-forward"
        self.layer_sizes = [10, 10, 10]
        self.layer_activations = ["Sigmoid"]
        self.loss_function_type = "mse"

        # for LSTM/Gru + Transformer
        self.num_hidden_layers = 1

        # for LSTM/Gru
        self.no_hidden_state = False
        self.bidirection = False

        # for transformer net
        self.dropout = 0.1
        self.num_heads = 10


class ParametersDescriptors(ParametersBase):
    """
    Parameters necessary for calculating/parsing input descriptors.

    Attributes
    ----------
    descriptor_type : string
        Type of descriptors that is used to represent the atomic fingerprint.
        Supported:

            - 'Bispectrum': Bispectrum descriptors (formerly called 'SNAP').
            - 'Atomic Density': Atomic density, calculated via Gaussian
                                descriptors.

    bispectrum_twojmax : int
        Bispectrum calculation: 2*jmax-parameter used for calculation of SNAP
        descriptors. Default value for jmax is 5, so default value for
        twojmax is 10.

    lammps_compute_file: string
        Bispectrum calculation: LAMMPS input file that is used to calculate the
        Bispectrum descriptors. If this string is empty, the standard LAMMPS input
        file found in this repository will be used (recommended).

    descriptors_contain_xyz : bool
        Legacy option. If True, it is assumed that the first three entries of
        the descriptor vector are the xyz coordinates and they are cut from the
        descriptor vector. If False, no such cutting is peformed.

    atomic_density_sigma : float
        Sigma used for the calculation of the Gaussian descriptors.
    """

    def __init__(self):
        super(ParametersDescriptors, self).__init__()
        self.descriptor_type = "Bispectrum"

        # These affect all descriptors, at least as long all descriptors
        # use LAMMPS (which they currently do).
        self.lammps_compute_file = ""
        self.descriptors_contain_xyz = True

        # TODO: I would rather handle the parallelization info automatically
        # and more under the hood. At this stage of the project this would
        # probably be overkill and hard to do, since there are many moving
        # parts, so for now let's keep this here, but in the future,
        # this should be adressed.
        self.use_z_splitting = True
        self.use_y_splitting = 0

        # Everything pertaining to the bispectrum descriptors.
        self.bispectrum_twojmax = 10
        self.bispectrum_cutoff = 4.67637
        self.bispectrum_switchflag = 1

        # Everything pertaining to the atomic density.
        # Seperate cutoff given here because bispectrum descriptors and
        # atomic density may be used at the same time, if e.g. bispectrum
        # descriptors are used for a full inference, which then uses the atomic
        # density for the calculation of the Ewald sum.
        self.use_atomic_density_energy_formula = False
        self.atomic_density_sigma = None
        self.atomic_density_cutoff = None

        # Everything concerning the minterpy descriptors.
        self.minterpy_point_list = []
        self.minterpy_cutoff_cube_size = 0.0
        self.minterpy_polynomial_degree = 4
        self.minterpy_lp_norm = 2

    @property
    def use_z_splitting(self):
        """
        Control whether splitting across the z-axis is used.

        Default is True, since this gives descriptors compatible with
        QE, for total energy evaluation. However, setting this value to False
        can, e.g. in the LAMMPS case, improve performance. This is relevant
        for e.g. preprocessing.
        """
        return self._use_z_splitting

    @use_z_splitting.setter
    def use_z_splitting(self, value):
        if value is False:
            self.use_y_splitting = 0
        self._use_z_splitting = value

    @property
    def use_y_splitting(self):
        """
        Control whether a splitting in y-axis is used.

        This can only be used in conjunction with a z-splitting, and
        the option will ignored if z-splitting is disabled. Only has an
        effect for values larger then 1.
        """
        return self._number_y_planes

    @use_y_splitting.setter
    def use_y_splitting(self, value):
        if self.use_z_splitting is False:
            self._number_y_planes = 0
        else:
            if value == 1:
                self._number_y_planes = 0
            else:
                self._number_y_planes = value

    @property
    def bispectrum_cutoff(self):
        """Cut off radius for bispectrum calculation."""
        return self._rcutfac

    @bispectrum_cutoff.setter
    def bispectrum_cutoff(self, value):
        self._rcutfac = value
        self.atomic_density_cutoff = value

    @property
    def bispectrum_switchflag(self):
        """
        Switchflag for the bispectrum calculation.

        Can only be 1 or 0. If 1 (default), a switching function will be used
        to ensure that atomic contributions smoothly go to zero after a
        certain cutoff. If 0 (old default, which can be problematic in some
        instances), this is not done, which can lead to discontinuities.
        """
        return self._snap_switchflag

    @bispectrum_switchflag.setter
    def bispectrum_switchflag(self, value):
        _int_value = int(value)
        if _int_value == 0:
            self._snap_switchflag = value
        if _int_value > 0:
            self._snap_switchflag = 1

    def _update_mpi(self, new_mpi):
        self._configuration["mpi"] = new_mpi

        # There may have been a serial or parallel run before that is now
        # no longer valid.
        self.lammps_compute_file = ""


class ParametersTargets(ParametersBase):
    """
    Parameters necessary for calculating/parsing output quantites.

    Attributes
    ----------
    target_type : string
        Number of points in the energy grid that is used to calculate the
        (L)DOS.

    ldos_gridsize : float
        Gridsize of the LDOS.

    ldos_gridspacing_ev: float
        Gridspacing of the energy grid the (L)DOS is evaluated on [eV].

    ldos_gridoffset_ev: float
        Lowest energy value on the (L)DOS energy grid [eV].

    pseudopotential_path : string
        Path at which pseudopotentials are located (for TEM).

    rdf_parameters : dict
        Parameters for calculating the radial distribution function(RDF).
        The RDF can directly be calculated via a function call, but if it is
        calculated e.g. during a MD or MC run, these parameters will control
        how. The following keywords are recognized:

        number_of_bins : int
            Number of bins used to create the histogram.

        rMax : float
            Radius up to which to calculate the RDF. None by default; this
            is the suggested behavior, as MALA will then on its own calculate
            the maximum radius up until which the calculation of the RDF is
            indisputably physically meaningful. Larger radii may be specified,
            e.g. for a Fourier transformation to calculate the static structure
            factor.

    tpcf_parameters : dict
        Parameters for calculating the three particle correlation function
        (TPCF).
        The TPCF can directly be calculated via a function call, but if it is
        calculated e.g. during a MD or MC run, these parameters will control
        how. The following keywords are recognized:

        number_of_bins : int
            Number of bins used to create the histogram.

        rMax : float
            Radius up to which to calculate the TPCF. If None, MALA will
            determine the maximum radius for which the TPCF is indisputably
            defined. Be advised - this may come at increased computational
            cost.

    ssf_parameters : dict
        Parameters for calculating the static structure factor
        (SSF).
        The SSF can directly be calculated via a function call, but if it is
        calculated e.g. during a MD or MC run, these parameters will control
        how. The following keywords are recognized:

        number_of_bins : int
            Number of bins used to create the histogram.

        kMax : float
            Maximum wave vector up to which to calculate the SSF.
    """

    def __init__(self):
        super(ParametersTargets, self).__init__()
        self.target_type = "LDOS"
        self.ldos_gridsize = 0
        self.ldos_gridspacing_ev = 0
        self.ldos_gridoffset_ev = 0
        self.restrict_targets = "zero_out_negative"
        self.pseudopotential_path = None
        self.rdf_parameters = {"number_of_bins": 500, "rMax": "mic"}
        self.tpcf_parameters = {"number_of_bins": 20, "rMax": "mic"}
        self.ssf_parameters = {"number_of_bins": 100, "kMax": 12.0}

    @property
    def restrict_targets(self):
        """
        Control if and how targets are restricted to physical values.

        Can be "zero_out_negative", i.e. all negative values are set to zero
        or "absolute_values", i.e. all negative values are multiplied by -1.
        """
        return self._restrict_targets

    @restrict_targets.setter
    def restrict_targets(self, value):
        if value != "zero_out_negative" and value != "absolute_values":
            self._restrict_targets = None
        else:
            self._restrict_targets = value


class ParametersData(ParametersBase):
    """
    Parameters necessary for loading and preprocessing data.

    Attributes
    ----------
    descriptors_contain_xyz : bool
        Legacy option. If True, it is assumed that the first three entries of
        the descriptor vector are the xyz coordinates and they are cut from the
        descriptor vector. If False, no such cutting is peformed.

    snapshot_directories_list : list
        A list of all added snapshots.

    data_splitting_type : string
        Specify how the data for validation, test and training is splitted.
        Currently the only supported option is by_snapshot,
        which splits the data by snapshot boundaries. It is also the default.

    input_rescaling_type : string
        Specifies how input quantities are normalized.
        Options:

            - "None": No normalization is applied.
            - "standard": Standardization (Scale to mean 0, standard
              deviation 1)
            - "normal": Min-Max scaling (Scale to be in range 0...1)
            - "feature-wise-standard": Row Standardization (Scale to mean 0,
              standard deviation 1)
            - "feature-wise-normal": Row Min-Max scaling (Scale to be in range
              0...1)

    output_rescaling_type : string
        Specifies how output quantities are normalized.
        Options:

            - "None": No normalization is applied.
            - "standard": Standardization (Scale to mean 0,
              standard deviation 1)
            - "normal": Min-Max scaling (Scale to be in range 0...1)
            - "feature-wise-standard": Row Standardization (Scale to mean 0,
              standard deviation 1)
            - "feature-wise-normal": Row Min-Max scaling (Scale to be in
              range 0...1)

    use_lazy_loading : bool
        If True, data is lazily loaded, i.e. only the snapshots that are
        currently needed will be kept in memory. This greatly reduces memory
        demands, but adds additional computational time.

    use_lazy_loading_prefetch : bool
        If True, will use alternative lazy loading path with prefetching
        for higher performance

    use_fast_tensor_data_set : bool
        If True, then the new, fast TensorDataSet implemented by Josh Romero
        will be used.

    shuffling_seed : int
        If not None, a seed that will be used to make the shuffling of the data
        in the DataShuffler class deterministic.
    """

    def __init__(self):
        super(ParametersData, self).__init__()
        self.snapshot_directories_list = []
        self.data_splitting_type = "by_snapshot"
        self.input_rescaling_type = "None"
        self.output_rescaling_type = "None"
        self.use_lazy_loading = False
        self.use_lazy_loading_prefetch = False
        self.use_fast_tensor_data_set = False
        self.shuffling_seed = None


class ParametersRunning(ParametersBase):
    """
    Parameters needed for network runs (train, test or inference).

    Some of these parameters only apply to either the train or test or
    inference case.

    Attributes
    ----------
    trainingtype : string
        Training type to be used. Supported options at the moment:

            - SGD: Stochastic gradient descent.
            - Adam: Adam Optimization Algorithm

    learning_rate : float
        Learning rate for chosen optimization algorithm. Default: 0.5.

    max_number_epochs : int
        Maximum number of epochs to train for. Default: 100.

    mini_batch_size : int
        Size of the mini batch for the optimization algorihm. Default: 10.

    weight_decay : float
        Weight decay for regularization. Always refers to L2 regularization.
        Default: 0.

    early_stopping_epochs : int
        Number of epochs the validation accuracy is allowed to not improve by
        at leastearly_stopping_threshold, before we terminate. If 0, no
        early stopping is performed. Default: 0.

    early_stopping_threshold : float
        Minimum fractional reduction in validation loss required to avoid
        early stopping, e.g. a value of 0.05 means that validation loss must
        decrease by 5% within early_stopping_epochs epochs or the training
        will be stopped early. More explicitly,
        validation_loss < validation_loss_old * (1-early_stopping_threshold)
        or the patience counter goes up.
        Default: 0. Numbers bigger than 0 can make early stopping very
        aggresive, while numbers less than 0 make the trainer very forgiving
        of loss increase.

    learning_rate_scheduler : string
        Learning rate scheduler to be used. If not None, an instance of the
        corresponding pytorch class will be used to manage the learning rate
        schedule.
        Options:

            - None: No learning rate schedule will be used.
            - "ReduceLROnPlateau": The learning rate will be reduced when the
              validation loss is plateauing.

    learning_rate_decay : float
        Decay rate to be used in the learning rate (if the chosen scheduler
        supports that).
        Default: 0.1

    learning_rate_patience : int
        Patience parameter used in the learning rate schedule (how long the
        validation loss has to plateau before the schedule takes effect).
        Default: 0.

    use_compression : bool
        If True and horovod is used, horovod compression will be used for
        allreduce communication. This can improve performance.

    num_workers : int
        Number of workers to be used for data loading.

    use_shuffling_for_samplers :
        If True, the training data will be shuffled in between epochs.
        If lazy loading is selected, then this shuffling will be done on
        a "by snapshot" basis.

    checkpoints_each_epoch : int
        If not 0, checkpoint files will be saved after eac
        checkpoints_each_epoch epoch.

    checkpoint_name : string
        Name used for the checkpoints. Using this, multiple runs
        can be performed in the same directory.

    visualisation : int
        If True then Tensorboard is activated for visualisation
        case 0: No tensorboard activated
        case 1: tensorboard activated with Loss and learning rate
        case 2; additonally weights and biases and gradient

    visualisation_dir : string
        Name of the folder that visualization files will be saved to.

    visualisation_dir_append_date : bool
        If True, then upon creating visualization files, these will be saved
        in a subfolder of visualisation_dir labelled with the starting date
        of the visualization, to avoid having to change input scripts often.

    inference_data_grid : list
        List holding the grid to be used for inference in the form of
        [x,y,z].

    use_mixed_precision : bool
        If True, mixed precision computation (via AMP) will be used.

    training_report_frequency : int
        Determines how often detailed performance info is printed during
        training (only has an effect if the verbosity is high enough).

    profiler_range : list
        List with two entries determining with which batch/iteration number
         the CUDA profiler will start and stop profiling. Please note that
         this option only holds significance if the nsys profiler is used.
    """

    def __init__(self):
        super(ParametersRunning, self).__init__()
        self.trainingtype = "SGD"
        self.learning_rate = 0.5
        self.max_number_epochs = 100
        self.verbosity = True
        self.mini_batch_size = 10
        self.weight_decay = 0
        self.early_stopping_epochs = 0
        self.early_stopping_threshold = 0
        self.learning_rate_scheduler = None
        self.learning_rate_decay = 0.1
        self.learning_rate_patience = 0
        self.use_compression = False
        self.num_workers = 0
        self.use_shuffling_for_samplers = True
        self.checkpoints_each_epoch = 0
        self.checkpoint_name = "checkpoint_mala"
        self.visualisation = 0
        self.visualisation_dir = os.path.join(".", "mala_logging")
        self.visualisation_dir_append_date = True
        self.during_training_metric = "ldos"
        self.after_before_training_metric = "ldos"
        self.inference_data_grid = [0, 0, 0]
        self.use_mixed_precision = False
        self.use_graphs = False
        self.training_report_frequency = 1000
        self.profiler_range = None #[1000, 2000]

    def _update_horovod(self, new_horovod):
        super(ParametersRunning, self)._update_horovod(new_horovod)
        self.during_training_metric = self.during_training_metric
        self.after_before_training_metric = self.after_before_training_metric

    @property
    def during_training_metric(self):
        """
        Control the metric used during training.

        Metric for evaluated on the validation set during training.
        Default is "ldos", meaning that the regular loss on the LDOS will be
        used as a metric. Possible options are "band_energy" and
        "total_energy". For these, the band resp. total energy of the
        validation snapshots will be calculated and compared to the provided
        DFT results. Of these, the mean average error in eV/atom will be
        calculated.
        """
        return self._during_training_metric

    @during_training_metric.setter
    def during_training_metric(self, value):
        if value != "ldos":
            if self._configuration["horovod"]:
                raise Exception("Currently, MALA can only operate with the "
                                "\"ldos\" metric for horovod runs.")
        self._during_training_metric = value

    @property
    def after_before_training_metric(self):
        """
        Get the metric used during training.

        Metric for evaluated on the validation and test set before and after
        training. Default is "LDOS", meaning that the regular loss on the LDOS
        will be used as a metric. Possible options are "band_energy" and
        "total_energy". For these, the band resp. total energy of the
        validation snapshots will be calculated and compared to the provided
        DFT results. Of these, the mean average error in eV/atom will be
        calculated.
        """
        return self._after_before_training_metric

    @after_before_training_metric.setter
    def after_before_training_metric(self, value):
        if value != "ldos":
            if self._configuration["horovod"]:
                raise Exception("Currently, MALA can only operate with the "
                                "\"ldos\" metric for horovod runs.")
        self._after_before_training_metric = value

    @during_training_metric.setter
    def during_training_metric(self, value):
        if value != "ldos":
            if self._configuration["horovod"]:
                raise Exception("Currently, MALA can only operate with the "
                                "\"ldos\" metric for horovod runs.")
        self._during_training_metric = value

    @property
    def use_graphs(self):
        """
        Decide whether CUDA graphs are used during training.

        Doing so will improve performance, but CUDA graphs are only available
        from CUDA 11.0 upwards.
        """
        return self._use_graphs

    @use_graphs.setter
    def use_graphs(self, value):
        if value is True:
            if self._configuration["gpu"] is False or \
                    torch.version.cuda is None:
                parallel_warn("No CUDA or GPU found, cannot use CUDA graphs.")
                value = False
            else:
                if float(torch.version.cuda) < 11.0:
                    raise Exception("Cannot use CUDA graphs with a CUDA"
                                    " version below 11.0")
        self._use_graphs = value


class ParametersHyperparameterOptimization(ParametersBase):
    """
    Hyperparameter optimization parameters.

    Attributes
    ----------
    direction : string
        Controls whether to minimize or maximize the loss function.
        Arguments are "minimize" and "maximize" respectively.

    n_trials : int
        Controls how many trials are performed (when using optuna).
        Default: 100.

    hlist : list
        List containing hyperparameters, that are then passed to optuna.
        Supported options so far include:

            - learning_rate (float): learning rate of the training algorithm
            - layer_activation_xxx (categorical): Activation function used for
              the feed forward network (see Netwok  parameters for supported
              activation functions). Note that _xxx is only so that optuna
              will differentiate between variables. No reordering is
              performed by the; the order depends on the order in the
              list. _xxx can be essentially anything. Please note further
              that you need to either only request one acitvation function
              (for all layers) or one for specifically for each layer.
            - ff_neurons_layer_xxx(int): Number of neurons per a layer. Note
              that _xxx is only so that optuna will differentiate between
              variables. No reordering is performed by MALA; the order
              depends on the order in the list. _xxx can be essentially
              anything.

        Users normally don't have to fill this list by hand, the hyperparamer
        optimizer provide interfaces for this task.


    hyper_opt_method : string
        Method used for hyperparameter optimization. Currently supported:

            - "optuna" : Use optuna for the hyperparameter optimization.
            - "oat" : Use orthogonal array tuning (currently limited to
              categorical hyperparemeters). Range analysis is
              currently done by simply choosing the lowest loss.
            - "naswot" : Using a NAS without training, based on jacobians.

    checkpoints_each_trial : int
        If not 0, checkpoint files will be saved after each
        checkpoints_each_trial trials. Currently, this only works with
        optuna.

    checkpoint_name : string
        Name used for the checkpoints. Using this, multiple runs
        can be performed in the same directory. Currently. this
        only works with optuna.

    study_name : string
        Name used for this study (in optuna#s storage). Necessary
        when operating with a RDB storage.

    rdb_storage : string
        Adress of the RDB storage to be used by optuna.

    rdb_storage_heartbeat : int
        Heartbeat interval for optuna (in seconds). Default is None.
        If not None and above 0, optuna will record the heartbeat of intervals.
        If no action on a RUNNING trial is recognized for longer then this
        interval, then this trial will be moved to FAILED. In distributed
        training, setting a heartbeat is currently the only way to achieve
        a precise number of trials:

        https://github.com/optuna/optuna/issues/1883

        For optuna versions below 2.8.0, larger heartbeat intervals are
        detrimental to performance and should be avoided:

        https://github.com/optuna/optuna/issues/2685

        For MALA, no evidence for decreased performance using smaller
        heartbeat values could be found. So if this is used, 1s is a reasonable
        value.

    number_training_per_trial : int
        Number of network trainings performed per trial. Default is 1,
        but it makes sense to choose a higher number, to exclude networks
        that performed by chance (good initilization). Naturally this impedes
        performance.

    trial_ensemble_evaluation : string
        Control how multiple trainings performed during a trial are evaluated.
        By default, simply "mean" is used. For smaller numbers of training
        per trial it might make sense to use "mean_std", which means that
        the mean of all metrics plus the standard deviation is used,
        as an estimate of the minimal accuracy to be expected. Currently,
        "mean" and "mean_std" are allowed.

    use_multivariate : bool
        If True, the optuna multivariate sampler is used. It is experimental
        since v2.2.0, but reported to perform very well.
        http://proceedings.mlr.press/v80/falkner18a.html

    naswot_pruner_cutoff : float
        If the surrogate loss algorithm is used as a pruner during a study,
        this cutoff determines which trials are neglected.

    pruner: string
        Pruner type to be used by optuna. Currently supported:

            - "multi_training": If multiple trainings are performed per
              trial, and one returns "inf" for the loss,
              no further training will be performed.
              Especially useful if used in conjunction
              with the band_energy metric.
            - "naswot": use the NASWOT algorithm as pruner

    naswot_pruner_batch_size : int
        Batch size for the NASWOT pruner

    number_bad_trials_before_stopping : int
        Only applies to optuna studies. If any integer above 0, then if no
        new best trial is found within number_bad_trials_before trials after
        the last one, the study will be stopped.

    sqlite_timeout : int
        Timeout for the SQLite backend of Optuna. This backend is officially
        not recommended because it is file based and can lead to errors;
        With a suitable timeout it can be used somewhat stable though and
        help in HPC settings.
    """

    def __init__(self):
        super(ParametersHyperparameterOptimization, self).__init__()
        self.direction = 'minimize'
        self.n_trials = 100
        self.hlist = []
        self.hyper_opt_method = "optuna"
        self.checkpoints_each_trial = 0
        self.checkpoint_name = "checkpoint_mala_ho"
        self.study_name = None
        self.rdb_storage = None
        self.rdb_storage_heartbeat = None
        self.number_training_per_trial = 1
        self.trial_ensemble_evaluation = "mean"
        self.use_multivariate = True
        self.naswot_pruner_cutoff = 0
        self.pruner = None
        self.naswot_pruner_batch_size = 0
        self.number_bad_trials_before_stopping = None
        self.sqlite_timeout = 600

        # For accelerated hyperparameter optimization.
        self.acsd_points = 100

    @property
    def rdb_storage_heartbeat(self):
        """Control whether a heartbeat is used for distributed optuna runs."""
        return self._rdb_storage_heartbeat

    @rdb_storage_heartbeat.setter
    def rdb_storage_heartbeat(self, value):
        if value == 0:
            self._rdb_storage_heartbeat = None
        else:
            self._rdb_storage_heartbeat = value

    @property
    def number_training_per_trial(self):
        """Control how many trainings are run per optuna trial."""
        return self._number_training_per_trial

    @number_training_per_trial.setter
    def number_training_per_trial(self, value):
        if value < 1:
            self._number_training_per_trial = 1
        else:
            self._number_training_per_trial = value

    @property
    def trial_ensemble_evaluation(self):
        """
        Control how multiple trainings performed during a trial are evaluated.

        By default, simply "mean" is used. For smaller numbers of training
        per trial it might make sense to use "mean_std", which means that
        the mean of all metrics plus the standard deviation is used,
        as an estimate of the minimal accuracy to be expected. Currently,
        "mean" and "mean_std" are allowed.
        """
        return self._trial_ensemble_evaluation

    @trial_ensemble_evaluation.setter
    def trial_ensemble_evaluation(self, value):
        if value != "mean" and value != "mean_std":
            self._trial_ensemble_evaluation = "mean"
        else:
            self._trial_ensemble_evaluation = value

    def show(self, indent=""):
        """
        Print name and values of all attributes of this object.

        Parameters
        ----------
        indent : string
            The indent used in the list with which the parameter
            shows itself.

        """
        for v in vars(self):
            if v != "_configuration":
                if v != "hlist":
                    if v[0] == "_":
                        printout(indent + '%-15s: %s' %
                                 (v[1:], getattr(self, v)), min_verbosity=0)
                    else:
                        printout(
                            indent + '%-15s: %s' % (v, getattr(self, v)),
                            min_verbosity=0)
                if v == "hlist":
                    i = 0
                    for hyp in self.hlist:
                        printout(indent + '%-15s: %s' %
                                 ("hyperparameter #"+str(i), hyp.name),
                                 min_verbosity=0)
                        i += 1


class ParametersDataGeneration(ParametersBase):
    """
    All parameters to help with data generation.

    Attributes
    ----------
    trajectory_analysis_denoising_width : int
        The distance metric is denoised prior to analysis using a certain
        width. This should be adjusted if there is reason to believe
        the trajectory will be noise for some reason.

    trajectory_analysis_below_average_counter : int
        Number of time steps that have to consecutively below the average
        of the distance metric curve, before we consider the trajectory
        to be equilibrated.
        Usually does not have to be changed.

    trajectory_analysis_estimated_equilibrium : float
        The analysis of the trajectory builds on the assumption that at some
        point of the trajectory, the system is equilibrated.
        For this, we need to provide the fraction of the trajectory (counted
        from the end). Usually, 10% is a fine assumption. This value usually
        does not need to be changed.

    trajectory_analysis_correlation_metric_cutoff : float
        Cutoff value to be used when sampling uncorrelated snapshots
        during trajectory analysis. If negative, a value will be determined
        numerically. This value is a cutoff for the minimum euclidean distance
        between any two ions in two subsequent ionic configurations.

    trajectory_analysis_temperature_tolerance_percent : float
        Maximum deviation of temperature between snapshot and desired
        temperature for snapshot to be considered for DFT calculation
        (in percent)

    local_psp_path : string
        Path to where the local pseudopotential is stored (for OF-DFT-MD).

    local_psp_name : string
        Name of the local pseudopotential (for OF-DFT-MD).

    ofdft_timestep : int
        Timestep of the OF-DFT-MD simulation.

    ofdft_number_of_timesteps : int
        Number of timesteps for the OF-DFT-MD simulation.

    ofdft_temperature : float
        Temperature at which to perform the OF-DFT-MD simulation.

    ofdft_kedf : string
        Kinetic energy functional to be used for the OF-DFT-MD simulation.

    ofdft_friction : float
        Friction to be added for the Langevin dynamics in the OF-DFT-MD run.
    """

    def __init__(self):
        super(ParametersDataGeneration, self).__init__()
        self.trajectory_analysis_denoising_width = 100
        self.trajectory_analysis_below_average_counter = 50
        self.trajectory_analysis_estimated_equilibrium = 0.1
        self.trajectory_analysis_correlation_metric_cutoff = -0.1
        self.trajectory_analysis_temperature_tolerance_percent = 1
        self.local_psp_path = None
        self.local_psp_name = None
        self.ofdft_timestep = 0
        self.ofdft_number_of_timesteps = 0
        self.ofdft_temperature = 0
        self.ofdft_kedf = "WT"
        self.ofdft_friction = 0.1


class Parameters:
    """
    All parameter MALA needs to perform its various tasks.

    Attributes
    ----------
    comment : string
        Characterizes a set of parameters (e.g. "experiment_ddmmyy").

    network : ParametersNetwork
        Contains all parameters necessary for constructing a neural network.

    descriptors : ParametersDescriptors
        Contains all parameters necessary for calculating/parsing descriptors.

    targets : ParametersTargets
        Contains all parameters necessary for calculating/parsing output
        quantites.

    data : ParametersData
        Contains all parameters necessary for loading and preprocessing data.

    running : ParametersRunning
        Contains parameters needed for network runs (train, test or inference).

    hyperparameters : ParametersHyperparameterOptimization
        Parameters used for hyperparameter optimization.

    debug : ParametersDebug
        Container for all debugging parameters.

    manual_seed: int
        If not none, this value is used as manual seed for the neural networks.
        Can be used to make experiments comparable. Default: None.
    """

    def __init__(self):
        self.comment = ""

        # Parameters subobjects.
        self.network = ParametersNetwork()
        self.descriptors = ParametersDescriptors()
        self.targets = ParametersTargets()
        self.data = ParametersData()
        self.running = ParametersRunning()
        self.hyperparameters = ParametersHyperparameterOptimization()
        self.datageneration = ParametersDataGeneration()

        # Attributes.
        self.manual_seed = None

        # Properties
        self.use_gpu = False
        self.use_horovod = False
        self.use_mpi = False
        self.verbosity = 1
        self.device = "cpu"
        self.openpmd_configuration = {}
        # TODO: Maybe as a percentage? Feature dimensions can be quite
        # different.
        self.openpmd_granularity = 1

    @property
    def openpmd_granularity(self):
        """
        Adjust the memory overhead of the OpenPMD interface.

        Smallest possible value is 1, meaning smallest memory footprint
        and slowest I/O. Higher values will introduce some memory penalty,
        but offer greater speed.
        The maximum level is the feature dimension of your data set, if
        you choose a value larger than this feature dimension, it will
        automatically be set to the feature dimension upon loading.
        """
        return self._openpmd_granularity

    @openpmd_granularity.setter
    def openpmd_granularity(self, value):
        if value < 1:
            value = 1
        self._openpmd_granularity = value
        self.network._update_openpmd_granularity(self._openpmd_granularity)
        self.descriptors._update_openpmd_granularity(self._openpmd_granularity)
        self.targets._update_openpmd_granularity(self._openpmd_granularity)
        self.data._update_openpmd_granularity(self._openpmd_granularity)
        self.running._update_openpmd_granularity(self._openpmd_granularity)
        self.hyperparameters._update_openpmd_granularity(self._openpmd_granularity)

    @property
    def verbosity(self):
        """
        Control the level of output for MALA.

        The following options are available:

            - 0: "low", only essential output will be printed
            - 1: "medium", most diagnostic output will be printed. (Default)
            - 2: "high", all information will be printed.


        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value
        set_current_verbosity(value)

    @property
    def use_gpu(self):
        """Control whether or not a GPU is used (provided there is one)."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        if value is False:
            self._use_gpu = False
        else:
            if torch.cuda.is_available():
                self._use_gpu = True
            else:
                parallel_warn("GPU requested, but no GPU found. MALA will "
                              "operate with CPU only.")

        # Invalidate, will be updated in setter.
        self.device = None
        self.network._update_gpu(self.use_gpu)
        self.descriptors._update_gpu(self.use_gpu)
        self.targets._update_gpu(self.use_gpu)
        self.data._update_gpu(self.use_gpu)
        self.running._update_gpu(self.use_gpu)
        self.hyperparameters._update_gpu(self.use_gpu)

    @property
    def use_horovod(self):
        """Control whether or not horovod is used for parallel training."""
        return self._use_horovod

    @use_horovod.setter
    def use_horovod(self, value):
        if value is False:
            self._use_horovod = False
        else:
            if horovod_available:
                hvd.init()
                # Invalidate, will be updated in setter.
                set_horovod_status(value)
                self.device = None
                self._use_horovod = value
                self.network._update_horovod(self.use_horovod)
                self.descriptors._update_horovod(self.use_horovod)
                self.targets._update_horovod(self.use_horovod)
                self.data._update_horovod(self.use_horovod)
                self.running._update_horovod(self.use_horovod)
                self.hyperparameters._update_horovod(self.use_horovod)
            else:
                parallel_warn("Horovod requested, but not installed found. "
                              "MALA will operate without horovod only.")


    @property
    def device(self):
        """Get the device used by MALA. Read-only."""
        return self._device

    @device.setter
    def device(self, value):
        device_id = get_local_rank()
        if self.use_gpu:
            self._device = "cuda:"\
                           f"{device_id}"
        else:
            self._device = "cpu"
        self.network._update_device(self._device)
        self.descriptors._update_device(self._device)
        self.targets._update_device(self._device)
        self.data._update_device(self._device)
        self.running._update_device(self._device)
        self.hyperparameters._update_device(self._device)

    @property
    def use_mpi(self):
        """Control whether or not horovod is used for parallel training."""
        return self._use_mpi

    @use_mpi.setter
    def use_mpi(self, value):
        set_mpi_status(value)
        # Invalidate, will be updated in setter.
        self.device = None
        self._use_mpi = value
        self.network._update_mpi(self.use_mpi)
        self.descriptors._update_mpi(self.use_mpi)
        self.targets._update_mpi(self.use_mpi)
        self.data._update_mpi(self.use_mpi)
        self.running._update_mpi(self.use_mpi)
        self.hyperparameters._update_mpi(self.use_mpi)

    @property
    def openpmd_configuration(self):
        """
        Provide a .toml or .json formatted string to configure OpenPMD.

        To load a configuration from a file, add an "@" in front of the file
        name and put the resulting string here. OpenPMD will then load
        the file. For further details, see the OpenPMD documentation.
        """
        return self._openpmd_configuration

    @openpmd_configuration.setter
    def openpmd_configuration(self, value):
        self._openpmd_configuration = value

        # Invalidate, will be updated in setter.
        self.network._update_openpmd_configuration(self.openpmd_configuration)
        self.descriptors._update_openpmd_configuration(self.openpmd_configuration)
        self.targets._update_openpmd_configuration(self.openpmd_configuration)
        self.data._update_openpmd_configuration(self.openpmd_configuration)
        self.running._update_openpmd_configuration(self.openpmd_configuration)
        self.hyperparameters._update_openpmd_configuration(self.openpmd_configuration)

    def show(self):
        """Print name and values of all attributes of this object."""
        printout("--- " + self.__doc__.split("\n")[1] + " ---",
                 min_verbosity=0)

        # Two for-statements so that global parameters are shown on top.
        for v in vars(self):
            if isinstance(getattr(self, v), ParametersBase):
                pass
            else:
                if v[0] == "_":
                    printout('%-15s: %s' % (v[1:], getattr(self, v)),
                             min_verbosity=0)
                else:
                    printout('%-15s: %s' % (v, getattr(self, v)),
                             min_verbosity=0)
        for v in vars(self):
            if isinstance(getattr(self, v), ParametersBase):
                parobject = getattr(self, v)
                printout("--- " + parobject.__doc__.split("\n")[1] + " ---",
                         min_verbosity=0)
                parobject.show("\t")

    def save(self, filename, save_format="json"):
        """
        Save the Parameters object to a file.

        Parameters
        ----------
        filename : string
            File to which the parameters will be saved to.

        save_format : string
            File format which is used for parameter saving.
            Currently only supported file format is "pickle".

        """
        if get_rank() != 0:
            return

        if save_format == "pickle":
            if filename[-3:] != "pkl":
                filename += ".pkl"
            with open(filename, 'wb') as handle:
                pickle.dump(self, handle, protocol=4)
        elif save_format == "json":
            if filename[-4:] != "json":
                filename += ".json"
            json_dict = {}
            members = inspect.getmembers(self,
                                         lambda a: not (inspect.isroutine(a)))

            # Two for loops so global properties enter the dict first.
            for member in members:
                # Filter out all private members, builtins, etc.
                if member[0][0] != "_" and member[0] != "device":
                    if isinstance(member[1], ParametersBase):
                        pass
                    else:
                        json_dict[member[0]] = member[1]
            for member in members:
                # Filter out all private members, builtins, etc.
                if member[0][0] != "_":
                    if isinstance(member[1], ParametersBase):
                        # All the subclasses have to provide this function.
                        member[1]: ParametersBase
                        json_dict[member[0]] = member[1].to_json()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_dict, f, ensure_ascii=False, indent=4)

        else:
            raise Exception("Unsupported parameter save format.")

    def save_as_pickle(self, filename):
        """
        Save the Parameters object to a pickle file.

        Parameters
        ----------
        filename : string
            File to which the parameters will be saved to.

        """
        self.save(filename, save_format="pickle")

    def save_as_json(self, filename):
        """
        Save the Parameters object to a json file.

        Parameters
        ----------
        filename : string
            File to which the parameters will be saved to.

        """
        self.save(filename, save_format="json")

    def optuna_singlenode_setup(self, wait_time=0):
        """
        Set up device and parallelization parameters for Optuna+MPI.

        This only needs to be called if multiple MPI ranks are used on
        one node to run Optuna. Optuna itself does NOT communicate via MPI.
        Thus, if we allocate e.g. one node with 4 GPUs and start 4 jobs,
        3 of those jobs will fail, because currently, we instantiate the
        cuda devices based on MPI ranks. This functions sets everything
        up properly. This of course requires MPI.
        This may be a bit hacky, but it lets us use one script and one
        MPI command to launch x GPU backed jobs on any node with x GPUs.

        Parameters
        ----------
        wait_time : int
            If larger than 0, then all processes will wait this many seconds
            times their rank number after this routine before proceeding.
            This can be useful when using a file based distribution algorithm.
        """
        # We first "trick" the parameters object to assume MPI and GPUs
        # are used. That way we get the right device.
        self.use_gpu = True
        self.use_mpi = True
        device_temp = self.device
        sleep(get_rank()*wait_time)

        # Now we can turn of MPI and set the device manually.
        self.use_mpi = False
        self._device = device_temp
        self.network._update_device(device_temp)
        self.descriptors._update_device(device_temp)
        self.targets._update_device(device_temp)
        self.data._update_device(device_temp)
        self.running._update_device(device_temp)
        self.hyperparameters._update_device(device_temp)

    @classmethod
    def load_from_file(cls, file, save_format="json",
                       no_snapshots=False):
        """
        Load a Parameters object from a file.

        Parameters
        ----------
        file : string or ZipExtFile
            File to which the parameters will be saved to.

        save_format : string
            File format which is used for parameter saving.
            Currently only supported file format is "pickle".

        no_snapshots : bool
            If True, than the snapshot list will be emptied. Useful when
            performing inference/testing after training a network.

        Returns
        -------
        loaded_parameters : Parameters
            The loaded Parameters object.

        """
        if save_format == "pickle":
            if isinstance(file, str):
                loaded_parameters = pickle.load(open(file, 'rb'))
            else:
                loaded_parameters = pickle.load(file)
            if no_snapshots is True:
                loaded_parameters.data.snapshot_directories_list = []
        elif save_format == "json":
            if isinstance(file, str):
                json_dict = json.load(open(file, encoding="utf-8"))
            else:
                json_dict = json.load(file)

            loaded_parameters = cls()
            for key in json_dict:
                if isinstance(json_dict[key], dict) and key \
                        != "openpmd_configuration":
                    # These are the other parameter classes.
                    sub_parameters =\
                        globals()[json_dict[key]["_parameters_type"]].\
                        from_json(json_dict[key])
                    setattr(loaded_parameters, key, sub_parameters)

            # We iterate a second time, to set global values, so that they
            # are properly forwarded.
            for key in json_dict:
                if not isinstance(json_dict[key], dict) or key == \
                        "openpmd_configuration":
                    setattr(loaded_parameters, key, json_dict[key])
            if no_snapshots is True:
                loaded_parameters.data.snapshot_directories_list = []
        else:
            raise Exception("Unsupported parameter save format.")

        return loaded_parameters

    @classmethod
    def load_from_pickle(cls, file, no_snapshots=False):
        """
        Load a Parameters object from a pickle file.

        Parameters
        ----------
        file : string or ZipExtFile
            File to which the parameters will be saved to.

        no_snapshots : bool
            If True, than the snapshot list will be emptied. Useful when
            performing inference/testing after training a network.

        Returns
        -------
        loaded_parameters : Parameters
            The loaded Parameters object.

        """
        return Parameters.load_from_file(file, save_format="pickle",
                                         no_snapshots=no_snapshots)

    @classmethod
    def load_from_json(cls, file, no_snapshots=False):
        """
        Load a Parameters object from a json file.

        Parameters
        ----------
        file : string or ZipExtFile
            File to which the parameters will be saved to.

        no_snapshots : bool
            If True, than the snapshot list will be emptied. Useful when
            performing inference/testing after training a network.

        Returns
        -------
        loaded_parameters : Parameters
            The loaded Parameters object.

        """
        return Parameters.load_from_file(file, save_format="json",
                                         no_snapshots=no_snapshots)
