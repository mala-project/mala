# Subclasses that make up the final parameters class.

class ParametersBase:
    def __init__(self):
        pass

    def show(self, indent=""):
        for v in vars(self):
            print(indent + '%-15s: %s' % (v, getattr(self, v)))


# noinspection PyMissingConstructor
class ParametersNetwork(ParametersBase):
    """Neural Network parameter subclass."""

    def __init__(self):
        super(ParametersNetwork, self).__init__()
        self.nn_type = "feed-forward"
        self.layer_sizes = [10,10,10]
        """Includes the input layer, although no activation is applied to it."""
        self.layer_activations = ["Sigmoid"]
        """If the dimension of layer_activations is smaller than the dimension of
        layer_sizes-1, than the first entry is used for all layers."""
        self.loss_function_type = "mse"


# noinspection PyMissingConstructor
class ParametersDescriptors(ParametersBase):
    """Fingerprint descriptor parameter subclass."""

    def __init__(self):
        super(ParametersDescriptors, self).__init__()
        self.descriptor_type = "SNAP"
        self.twojmax = 10
        """
        SNAP calculation: 2*jmax-parameter used for calculation of SNAP descriptors. Standard value
        for jmax is 5, so standard value for twojmax is 10
        """
        self.rcutfac = 4.67637
        """
        SNAP calculation: radius cutoff factor for the fingerprint sphere in Angstroms.
        """
        self.lammps_compute_file = ""
        """
        SNAP calculation: LAMMPS input file that is used to calculate the SNAP descriptors.
        If this string is empty, the standard LAMMPS input file found in this
        repository will be used (recommended).
        """


# noinspection PyMissingConstructor
class ParametersTargets(ParametersBase):
    """Target quantity parsing parameter subclass."""

    def __init__(self):
        super(ParametersTargets, self).__init__()
        self.target_type = "LDOS"
        self.ldos_gridsize = 0
        """
        Number of points in the energy grid that is used to calculate the LDOS.
        """


# noinspection PyMissingConstructor
class ParametersData(ParametersBase):
    """Dataset interface parameter subclass."""

    def __init__(self):
        super(ParametersData, self).__init__()
        self.datatype_in = "*.npy"
        """
        Specifies the kind of input data we are working with.
            Implemented so far:
                - mnist for the MNIST data set (for testing purposes).
                - qe.out for QuantumEspresso out files.
                - *.npy for preprocessed input
        """
        self.descriptors_contain_xyz = True
        """
        Legacy option. If true, it is assumed that the first three entries of the descriptor vector are the xyz 
        coordinates and they are cut from the descriptor vector. If False, no such cutting is peformed.
        """
        self.input_memmap_mode = None
        """
        Numpy memmap option used for all the numpy.save operations on the input data.
        """
        self.datatype_out = "*.npy"
        """
        Specifies the kind of input data we are working with.
            Implemented so far:
                - mnist for the MNIST data set (for testing purposes).
                - *.cube for cube files containing e.g. the LDOS.
                - *.npy for preprocessed output
        """
        self.output_memmap_mode = None
        """
        Numpy memmap option used for all the numpy.save operations on the output data.
        """
        self.snapshot_directories_list = []
        """
        A list of all added snapshots.
        """
        self.data_splitting_type = "random"
        """Specify how the data for validation, test and training is splitted.
        Currently implemented:
            - random: split the data randomly, ignore snapshot boundaries.
            - by_snapshot: split the data by snapshot boundaries.
        """
        self.data_splitting_percent = [0,0,0]
        """
        Details how much of the data is used for training, validation and testing [%].
        """
        self.data_splitting_snapshots = ["te", "te", "te"]
        """
        Details how (and which!) snapshots are used for what [#snapshots]:
            - te: This snapshot will be a testing snapshot.
            - tr: This snapshot will be a training snapshot.
            - va: This snapshot will be a validation snapshot.
        Please note that the length of this list and the number of snapshots must be identical.
        """
        self.input_rescaling_type = "None"
        """
        Specifies how input quantities are normalized.
        Options:
            - "None": No normalization is applied.
            - "standard": Standardization (Scale to mean 0, standard deviation 1)
            - "normal": Min-Max scaling (Scale to be in range 0...1)
            - "feature-wise-standard": Row Standardization (Scale to mean 0, standard deviation 1)
            - "feature-wise-normal": Row Min-Max scaling (Scale to be in range 0...1)
        """
        self.output_rescaling_type = "None"
        """
        Specifies how output quantities are normalized.
        Options:
            - "None": No normalization is applied.
            - "standard": Standardization (Scale to mean 0, standard deviation 1)
            - "normal": Min-Max scaling (Scale to be in range 0...1)
            - "feature-wise-standard": Row Standardization (Scale to mean 0, standard deviation 1)
            - "feature-wise-normal": Row Min-Max scaling (Scale to be in range 0...1)
        """


class ParametersTraining(ParametersBase):
    """Network training parameter subclass."""

    def __init__(self):
        super(ParametersTraining, self).__init__()
        self.trainingtype = "SGD"
        """Training type to be used. Options at the moment:
            - SGD: Stochastic gradient descent.
            - Adam: Adam Optimization Algorithm 
        """
        self.learning_rate = 0.5
        """
        Learning rate for chosen optimization algorithm.
        """
        self.max_number_epochs = 100
        """
        Maximum number of epochs we train for.
        """
        # TODO: Find a better system for verbosity. Maybe a number.
        self.verbosity = True
        """
        Determines if training output is shown during training.
        """
        self.mini_batch_size = 10
        """
        Size of the mini batch for the optimization algorihm.
        """
        self.weight_decay = 0
        """
        Weight decay for regularization. Always refers to L2 regularization.
        """
        self.early_stopping_epochs = 0
        """
        Number of epochs the validation accuracy is allowed to not improve by at leastearly_stopping_threshold, before we
        terminate. If 0, no early stopping is performed.
        """
        self.early_stopping_threshold = 0
        """
        If the validation accuracy does not improve by at least threshold for early_stopping_epochs epochs, training
        is terminated:
        validation_loss < validation_loss_old * (1+early_stopping_threshold), or patience counter will go up.
        """
        self.learning_rate_scheduler = None
        """
        Learning rate scheduler to be used. If not None, an instance of the corresponding pytorch class will be 
        used to manage the learning rate schedule.
        Options:
            - None: No learning rate schedule will be used.
            - ReduceLROnPlateau: The learning rate will be reduced when the validation loss is plateauing. 
        """
        self.learning_rate_decay = 0.1
        """
        Decay rate to be used in the learning rate (if the chosen scheduler supports that).
        """
        self.learning_rate_patience = 0
        """
        Patience parameter used in the learning rate schedule (how long the validation loss has to plateau before
        the schedule takes effect).
        """


class ParametersHyperparameterOptinization(ParametersBase):
    """Hyperparameter optimization subclass."""

    def __init__(self):
        super(ParametersHyperparameterOptinization, self).__init__()
        self.direction = 'minimize'
        """
        Controls whether we minimize or maximize the loss function.
        """
        self.n_trials = 100
        """
        Controls how many trials optuna performs.
        """
        self.hlist = []
        """
        List containing hyperparameters, that are then passed to optuna. 
        Supported options so far include:
            learning_rate (float): learning rate of the training algorithm
            layer_activation_xxx (categorical): Activation function used for the feed forward network (see Netwok 
                                            parameters for supported activation functions). Note that _xxx is only so 
                                            that optuna will differentiate between variables. No reordering is performed
                                             by our code; the order depends on the order in the list. _xxx can be 
                                             essentially anything. Please note further that you need to either only 
                                             request one acitvation function (for all layers) or one for specifically 
                                             for each layer.
            ff_neurons_layer_xxx(int): Number of neurons per a layer. Note that _xxx is only so that optuna will 
                                        differentiate between variables. No reordering is performed by our code;
                                        the order depends on the order in the list. _xxx can be essentially anything.
        """
    def show(self, indent=""):
        for v in vars(self):
            if v != "hlist":
                print(indent + '%-15s: %s' % (v, getattr(self, v)))
            if v == "hlist":
                i = 0
                for hyp in self.hlist:
                    print(indent + '%-15s: %s' % ("hyperparameter #"+str(i), hyp.name))
                    i += 1


class ParametersDebug(ParametersBase):
    """Central debugging parameters. Can be used
    to e.g. reduce number of data."""

    def __init__(self):
        super(ParametersDebug, self).__init__()
        self.grid_dimensions = []
        """
        Enforces a smaller grid size globally.
        """


# TODO: Add keyword arguments that allow for a passing of the arguments in the constructor.
class Parameters:
    """Parameter class for ML-DFT@CASUS."""

    def __init__(self):
        self.comment = 0
        '''Comment a used parameter dataset.'''
        self.network = ParametersNetwork()
        self.descriptors = ParametersDescriptors()
        self.targets = ParametersTargets()
        self.data = ParametersData()
        self.training = ParametersTraining()
        self.hyperparameters = ParametersHyperparameterOptinization()
        self.debug = ParametersDebug()

    def show(self):
        """Prints all the parameters bundled in this class."""
        print("--- " + self.__doc__ + " ---")
        for v in vars(self):
            if isinstance(getattr(self, v), ParametersBase):
                parobject = getattr(self, v)
                print("--- " + parobject.__doc__ + " ---")
                parobject.show("\t")
            else:
                print('%-15s: %s' % (v, getattr(self, v)))


if __name__ == "__main__":
    p = Parameters()
    p.show()
    print("parameters.py - basic functions are working.")
