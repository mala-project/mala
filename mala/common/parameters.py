"""Collection of all parameter related classes and functions."""
import pickle
import warnings
from .printout import printout, set_horovod_status
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    warnings.warn("You either don't have Horovod installed or it is not "
                  "configured correctly. You can still train networks, but "
                  "attempting to set parameters.training.use_horovod = "
                  "True WILL cause a crash.", stacklevel=3)
import torch


class ParametersBase:
    """Base parameter class for MALA."""

    def __init__(self):
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
            printout(indent + '%-15s: %s' % (v, getattr(self, v)))


class ParametersNetwork(ParametersBase):
    """
    Parameters necessary for constructing a neural network.

    Attributes
    ----------
    nn_type : string
        Type of the neural network that will be used. Currently supported is
        only feed-forward, which is also the default

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
    """

    def __init__(self):
        super(ParametersNetwork, self).__init__()
        self.nn_type = "feed-forward"
        self.layer_sizes = [10, 10, 10]
        self.layer_activations = ["Sigmoid"]
        self.loss_function_type = "mse"


class ParametersDescriptors(ParametersBase):
    """
    Parameters necessary for calculating/parsing input descriptors.

    Attributes
    ----------
    descriptor_type : string
        Type of descriptors that is used to represent the atomic fingerprint.
        Currently only "SNAP" is supported.

    twojmax : int
        SNAP calculation: 2*jmax-parameter used for calculation of SNAP
        descriptors. Default value for jmax is 5, so default value for
        twojmax is 10.

    rcutfac: float
        SNAP calculation: radius cutoff factor for the fingerprint sphere in
        Angstroms. Default value is 4.67637.

    lammps_compute_file: string
        SNAP calculation: LAMMPS input file that is used to calculate the
        SNAP descriptors. If this string is empty, the standard LAMMPS input
        file found in this repository will be used (recommended).

    """

    def __init__(self):
        super(ParametersDescriptors, self).__init__()
        self.descriptor_type = "SNAP"
        self.twojmax = 10
        self.rcutfac = 4.67637
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
        Gridspacing of the energy grid the (L)DOS is evaluated on [eV].

    ldos_gridspacing_ev: float
        SNAP calculation: radius cutoff factor for the fingerprint sphere in
        Angstroms. Default value is 4.67637.

    ldos_gridoffset_ev: float
        Lowest energy value on the (L)DOS energy grid [eV].
    """

    def __init__(self):
        super(ParametersTargets, self).__init__()
        self.target_type = "LDOS"
        self.ldos_gridsize = 0
        self.ldos_gridspacing_ev = 0
        self.ldos_gridoffset_ev = 0


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

    data_splitting_snapshots : list
        Details how (and which!) snapshots are used for what:

          - te: This snapshot will be a testing snapshot.
          - tr: This snapshot will be a training snapshot.
          - va: This snapshot will be a validation snapshot.

        Please note that the length of this list and the number of snapshots
        must be identical. The first element of this list will be used
        to characterize the first snapshot, the second element for the second
        snapshot etc.

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
    """

    def __init__(self):
        super(ParametersData, self).__init__()
        self.descriptors_contain_xyz = True
        self.snapshot_directories_list = []
        self.data_splitting_type = "by_snapshot"
        # self.data_splitting_percent = [0,0,0]
        self.data_splitting_snapshots = []
        self.input_rescaling_type = "None"
        self.output_rescaling_type = "None"
        self.use_lazy_loading = False


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

    verbosity : bool
        If True, training output is shown during training. Default: True.

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
        If the validation accuracy does not improve by at least threshold for
        early_stopping_epochs epochs, training is terminated:
        validation_loss < validation_loss_old * (1+early_stopping_threshold),
        or patience counter will go up.
        Default: 0. Numbers bigger than 0 can make early stopping very
        aggresive.

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

    kwargs : dict
        Dictionary for keyword arguments for horovod.

    sampler : dict
        Dictionary with samplers.

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
    """

    def __init__(self):
        super(ParametersRunning, self).__init__()
        self.trainingtype = "SGD"
        self.learning_rate = 0.5
        self.max_number_epochs = 100
        # TODO: Find a better system for verbosity. Maybe a number.
        self.verbosity = True
        self.mini_batch_size = 10
        self.weight_decay = 0
        self.early_stopping_epochs = 0
        self.early_stopping_threshold = 0
        self.learning_rate_scheduler = None
        self.learning_rate_decay = 0.1
        self.learning_rate_patience = 0
        self.use_compression = False
        # TODO: Give this parameter a more descriptive name.
        self.kwargs = {'num_workers': 0, 'pin_memory': False}
        # TODO: Objects should not be parameters!
        self.sampler = {"train_sampler": None, "validate_sampler": None,
                        "test_sampler": None}
        self.use_shuffling_for_samplers = True
        self.checkpoints_each_epoch = 0
        self.checkpoint_name = "checkpoint_mala"


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
              currently done by simply choosing the lowesr loss.
            - "notraining" : Using a NAS without training, based on jacobians.

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
            if v != "hlist":
                printout(indent + '%-15s: %s' % (v, getattr(self, v)))
            if v == "hlist":
                i = 0
                for hyp in self.hlist:
                    printout(indent + '%-15s: %s' %
                             ("hyperparameter #"+str(i), hyp.name))
                    i += 1


class ParametersDebug(ParametersBase):
    """
    All debugging parameters.

    Attributes
    ----------
    grid_dimensions : list
        A list containing three elements. It enforces a smaller grid size
        globally when it is not empty.. Default : []

    """

    def __init__(self):
        super(ParametersDebug, self).__init__()
        self.grid_dimensions = []


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
        self.network = ParametersNetwork()
        self.descriptors = ParametersDescriptors()
        self.targets = ParametersTargets()
        self.data = ParametersData()
        self.running = ParametersRunning()
        self.hyperparameters = ParametersHyperparameterOptimization()
        self.debug = ParametersDebug()
        self.manual_seed = None

        # Properties
        self.use_horovod = False
        self.use_gpu = False

    @property
    def use_gpu(self):
        """Control whether or not a GPU is used (provided there is one)."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        if value is False:
            self._use_gpu = False
        if value is True:
            if torch.cuda.is_available():
                self._use_gpu = True
            else:
                warnings.warn("GPU requested, but no GPU found. MALA will "
                              "operate with CPU only.", stacklevel=3)

    @property
    def use_horovod(self):
        """Control whether or not horovod is used for parallel training."""
        return self._use_horovod

    @use_horovod.setter
    def use_horovod(self, value):
        if value:
            hvd.init()
        set_horovod_status(value)
        self._use_horovod = value

    def show(self):
        """Print name and values of all attributes of this object."""
        printout("--- " + self.__doc__.split("\n")[1] + " ---")
        for v in vars(self):
            if isinstance(getattr(self, v), ParametersBase):
                parobject = getattr(self, v)
                printout("--- " + parobject.__doc__.split("\n")[1] + " ---")
                parobject.show("\t")
            else:
                printout('%-15s: %s' % (v, getattr(self, v)))

    def save(self, filename, save_format="pickle"):
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
        if self.use_horovod:
            if hvd.rank() != 0:
                return
        if save_format == "pickle":
            with open(filename, 'wb') as handle:
                pickle.dump(self, handle, protocol=4)
        else:
            raise Exception("Unsupported parameter save format.")

    @classmethod
    def load_from_file(cls, filename, save_format="pickle",
                       no_snapshots=False):
        """
        Load a Parameters object from a file.

        Parameters
        ----------
        filename : string
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
            with open(filename, 'rb') as handle:
                loaded_parameters = pickle.load(handle)
                if no_snapshots is True:
                    loaded_parameters.data.snapshot_directories_list = []
        else:
            raise Exception("Unsupported parameter save format.")

        return loaded_parameters
