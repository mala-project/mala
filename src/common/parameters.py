# Subclasses that make up the final parameters class.

class ParametersBase:
    def __init__(self):
        raise Exception("Error: No constructor implemented.")

    def show(self, indent=""):
        for v in vars(self):
            print(indent + '%-15s: %s' % (v, getattr(self, v)))


# noinspection PyMissingConstructor
class ParametersNetwork(ParametersBase):
    """Neural Network parameter subclass."""

    def __init__(self):
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
        self.target_type = "LDOS"
        self.ldos_gridsize = 0
        """
        Number of points in the energy grid that is used to calculate the LDOS.
        """


# noinspection PyMissingConstructor
class ParametersData(ParametersBase):
    """Dataset interface parameter subclass."""

    def __init__(self):
        self.datatype_in = "qe.out"
        """
        Specifies the kind of input data we are working with.
            Implemented so far:
                - mnist for the MNIST data set (for testing purposes).
                - qe.out for QuantumEspresso out files.
        """
        self.datatype_out = "*.cube"
        """
        Specifies the kind of input data we are working with.
            Implemented so far:
                - mnist for the MNIST data set (for testing purposes).
                - *.cube for cube files containing e.g. the LDOS.
        """
        self.snapshot_directories_list = []
        """
        A list of all added snapshots.
        """
        self.data_splitting_type = "random"
        """Specify how the data for validation, test and training is splitted.
        Currently implemented:
            - random: split the data randomly, ignore snapshot boundaries.
        """
        self.data_splitting_percent = [0,0,0]
        """
        Details how much of the data is used for training, validation and testing [%].
        """
        self.data_splitting_snapshots = [0,0,0]
        """
        Details how much of the data is used for training, validation and testing [#snapshots].
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
        self.trainingtype = "SGD"
        """Training type to be used."""
        self.learning_rate = 0.5
        self.max_number_epochs = 100
        # TODO: Find a better system for verbosity. Maybe a number.
        self.verbosity = True
        self.mini_batch_size = 10


class ParametersHyperparameterOptinization(ParametersBase):
    """Hyperparameter optimization subclass."""

    def __init__(self):
        self.direction = 'minimize'
        """
        Controls whether we minimize or maximize the loss function.
        """
        self.n_trials = 100
        """
        Controls how many trials optuna performs.
        """
        # self.optimization_list
        # """
        #
        # """

# noinspection PyMissingConstructor
class ParametersDebug(ParametersBase):
    """Central debugging parameters. Can be used
    to e.g. reduce number of data."""

    def __init__(self):
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
            if (isinstance(getattr(self, v), ParametersBase)):
                parobject = getattr(self, v)
                print("--- " + parobject.__doc__ + " ---")
                parobject.show("\t")
            else:
                print('%-15s: %s' % (v, getattr(self, v)))


if __name__ == "__main__":
    p = Parameters()
    p.show()
    print("parameters.py - basic functions are working.")
