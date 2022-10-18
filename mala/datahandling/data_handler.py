"""DataHandler class that loads and scales data."""
import os

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch
from torch.utils.data import TensorDataset

from mala.common.parallelizer import printout, barrier
from mala.common.parameters import Parameters, ParametersData
from mala.datahandling.data_scaler import DataScaler
from mala.datahandling.snapshot import Snapshot
from mala.datahandling.lazy_load_dataset import LazyLoadDataset
from mala.datahandling.lazy_load_dataset_clustered import LazyLoadDatasetClustered
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target


class DataHandler:
    """
    Loads and scales data. Can only process numpy arrays at the moment.

    Data that is not in a numpy array can be converted using the DataConverter
    class.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data. If None, then one will
        be created by this class.

    target_calculator : mala.targets.target.Target
        Used to do unit conversion on output data. If None, then one will
        be created by this class.

    input_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the input data. If None, then one will be created by
        this class.

    output_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the output data. If None, then one will be created by
        this class.
    """

    def __init__(self, parameters: Parameters, target_calculator=None,
                 descriptor_calculator=None, input_data_scaler=None,
                 output_data_scaler=None):
        self.parameters: ParametersData = parameters.data
        self.dbg_grid_dimensions = parameters.debug.grid_dimensions
        self.use_horovod = parameters.use_horovod
        self.training_data_set = None

        self.validation_data_set = None

        self.test_data_set = None

        self.input_data_scaler = input_data_scaler
        if self.input_data_scaler is None:
            self.input_data_scaler \
                = DataScaler(self.parameters.input_rescaling_type,
                             use_horovod=self.use_horovod)

        self.output_data_scaler = output_data_scaler
        if self.output_data_scaler is None:
            self.output_data_scaler \
                = DataScaler(self.parameters.output_rescaling_type,
                             use_horovod=self.use_horovod)

        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptor(parameters)

        self.nr_snapshots = 0
        self.grid_dimension = [0, 0, 0]
        self.grid_size = 0

        # Actual data points in the different categories.
        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0

        # Number of snapshots in these categories.
        self.nr_training_snapshots = 0
        self.nr_test_snapshots = 0
        self.nr_validation_snapshots = 0

        self.training_data_inputs = torch.empty(0)
        """
        Torch tensor holding all scaled training data inputs.
        """

        self.validation_data_inputs = torch.empty(0)
        """
        Torch tensor holding all scaled validation data inputs.
        """

        self.test_data_inputs = torch.empty(0)
        """
        Torch tensor holding all scaled testing data inputs.
        """

        self.training_data_outputs = torch.empty(0)
        """
        Torch tensor holding all scaled training data output.
        """

        self.validation_data_outputs = torch.empty(0)
        """
        Torch tensor holding all scaled validation data output.
        """

        self.test_data_outputs = torch.empty(0)
        """
        Torch tensor holding all scaled testing data output.
        """

    def get_input_dimension(self):
        """
        Get the dimension of the input vector.

        Returns
        -------
        input_dimension : int
            Dimension of the input vector.
        """
        return self.input_dimension

    def get_output_dimension(self):
        """
        Get the dimension of the output vector.

        Returns
        -------
        output_dimension : int
            Dimension of the output vector.
        """
        return self.output_dimension

    def add_snapshot(self, input_npy_file, input_npy_directory,
                     output_npy_file, output_npy_directory, add_snapshot_as,
                     output_units="1/(eV*A^3)", input_units="None",
                     calculation_output_file=""):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        input_npy_file : string
            File with saved numpy input array.

        input_npy_directory : string
            Directory containing input_npy_directory.

        output_npy_file : string
            File with saved numpy output array.

        output_npy_directory : string
            Directory containing output_npy_file.

        input_units : string
            Units of input data. See descriptor classes to see which units are
            supported.

        output_units : string
            Units of output data. See target classes to see which units are
            supported.

        calculation_output_file : string
            File with the output of the original snapshot calculation. This is
            only needed when testing multiple snapshots.

        add_snapshot_as : string
            Must be "tr", "va" or "te", the snapshot will be added to the
            snapshot list as training, validation or testing snapshot,
            respectively.
        """
        snapshot = Snapshot(input_npy_file, input_npy_directory,
                            output_npy_file, output_npy_directory,
                            add_snapshot_as,
                            input_units=input_units,
                            output_units=output_units,
                            calculation_output=calculation_output_file)
        self.parameters.snapshot_directories_list.append(snapshot)

    def clear_data(self):
        """
        Reset the entire data pipeline.

        Useful when doing multiple investigations in the same python file.
        """
        self.training_data_set = None
        self.validation_data_set = None
        self.test_data_set = None
        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0
        self.nr_training_snapshots = 0
        self.nr_test_snapshots = 0
        self.nr_validation_snapshots = 0
        self.parameters.snapshot_directories_list = []

    def prepare_data(self, reparametrize_scaler=True):
        """
        Prepare the data to be used in a training process.

        This includes:

            - Checking snapshots for consistency
            - Parametrizing the DataScalers (if desired)
            - Building DataSet objects.

        Parameters
        ----------
        reparametrize_scaler : bool
            If True (default), the DataScalers are parametrized based on the
            training data.

        """
        # Do a consistency check of the snapshots so that we don't run into
        # an error later. If there is an error, check_snapshots() will raise
        # an exception.
        printout("Checking the snapshots and your inputs for consistency.",
                 min_verbosity=1)
        self.__check_snapshots()
        printout("Consistency check successful.", min_verbosity=0)

        # If the DataHandler is used for inference, i.e. no training or
        # validation snapshots have been provided,
        # than we can definitely not reparametrize the DataScalers.
        if self.nr_training_data == 0:
            reparametrize_scaler = False
            if self.input_data_scaler.cantransform is False or \
                    self.output_data_scaler.cantransform is False:
                raise Exception("In inference mode, the DataHandler needs "
                                "parametrized DataScalers, "
                                "while you provided unparametrized "
                                "DataScalers.")

        # Parametrize the scalers, if needed.
        if reparametrize_scaler:
            printout("Initializing the data scalers.", min_verbosity=1)
            self.__parametrize_scalers()
            printout("Data scalers initialized.", min_verbosity=0)
        else:
            printout("Data scalers already initilized, loading data to RAM.",
                     min_verbosity=0)
            if self.parameters.use_lazy_loading is False:
                self.__load_training_data_into_ram()

        # Build Datasets.
        printout("Build datasets.", min_verbosity=1)
        self.__build_datasets()
        printout("Build dataset: Done.", min_verbosity=0)

        # Wait until all ranks are finished with data preparation.
        # It is not uncommon that ranks might be asynchronous in their
        # data preparation by a small amount of minutes. If you notice
        # an elongated wait time at this barrier, check that your file system
        # allows for parallel I/O.
        barrier()

    def mix_datasets(self):
        """
        For lazily-loaded data sets, the snapshot ordering is (re-)mixed.

        This applies only to the training data set. For the validation and
        test set it does not matter.
        """
        if self.parameters.use_lazy_loading:
            self.training_data_set.mix_datasets()

    def raw_numpy_to_converted_scaled_tensor(self, numpy_array, data_type,
                                             units, convert3Dto1D=False):
        """
        Transform a raw numpy array into a scaled torch tensor.

        This tensor will also be in the right units, i.e. a tensor that can
        simply be put into a MALA network.

        Parameters
        ----------
        numpy_array : np.array
            Array that is to be converted.
        data_type : string
            Either "in" or "out", depending if input or output data is
            processed.
        units : string
            Units of the data that is processed.
        convert3Dto1D : bool
            If True (default: False), then a (x,y,z,dim) array is transformed
            into a (x*y*z,dim) array.

        Returns
        -------
        converted_tensor: torch.Tensor
            The fully converted and scaled tensor.
        """
        # Check parameters for consistency.
        if data_type != "in" and data_type != "out":
            raise Exception("Please specify either \"in\" or \"out\" as "
                            "data_type.")

        # Convert units of numpy array.
        numpy_array = self.__raw_numpy_to_converted_numpy(numpy_array,
                                                          data_type, units)

        # If desired, the dimensions can be changed.
        if convert3Dto1D:
            if data_type == "in":
                data_dimension = self.get_input_dimension()
            else:
                data_dimension = self.get_output_dimension()
            desired_dimensions = [self.grid_size, data_dimension]
        else:
            desired_dimensions = None

        # Convert numpy array to scaled tensor a network can work with.
        numpy_array = self.\
            __converted_numpy_to_scaled_tensor(numpy_array, desired_dimensions,
                                               data_type)
        return numpy_array

    def resize_snapshots_for_debugging(self, directory="./",
                                       naming_scheme_input=
                                       "test_Al_debug_2k_nr*.in",
                                       naming_scheme_output=
                                       "test_Al_debug_2k_nr*.out"):
        """
        Resize all snapshots in the list.

        Parameters
        ----------
        directory : string
            Directory to which the resized snapshots should be saved.

        naming_scheme_input : string
            Naming scheme for the resulting input numpy files.

        naming_scheme_output : string
            Naming scheme for the resulting output numpy files.

        """
        i = 0
        snapshot: Snapshot
        for snapshot in self.parameters.snapshot_directories_list:
            tmp_array = self.__load_from_npy_file(
                os.path.join(snapshot.input_npy_directory,
                             snapshot.input_npy_file))
            tmp_file_name = naming_scheme_input
            tmp_file_name = tmp_file_name.replace("*", str(i))
            np.save(os.path.join(directory, tmp_file_name) + ".npy", tmp_array)

            tmp_array = self.__load_from_npy_file(
                os.path.join(snapshot.output_npy_directory,
                             snapshot.output_npy_file))
            tmp_file_name = naming_scheme_output
            tmp_file_name = tmp_file_name.replace("*", str(i))
            np.save(os.path.join(directory, tmp_file_name + ".npy"), tmp_array)
            i += 1

    def get_snapshot_calculation_output(self, snapshot_number):
        """
        Get the path to the output file for a specific snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the calculation output should be returned.

        Returns
        -------
        calculation_output : string
            Path to the calculation output for this snapshot.

        """
        return self.parameters.snapshot_directories_list[snapshot_number].\
            calculation_output

    def prepare_for_testing(self):
        """
        Prepare DataHandler for usage within Tester class.

        Ensures that lazily-loaded data sets do not perform unnecessary I/O
        operations. Only needed in Tester class.
        """
        if self.parameters.use_lazy_loading:
            self.test_data_set.return_outputs_directly = True

    def get_test_input_gradient(self, snapshot_number):
        """
        Get the gradient of the test inputs for an entire snapshot.

        This gradient will be returned as scaled Tensor.
        The reason the gradient is returned (rather then returning the entire
        inputs themselves) is that by slicing a variable, pytorch no longer
        considers it a "leaf" variable and will stop tracking and evaluating
        its gradient. Thus, it is easier to obtain the gradient and then
        slice it.

        Parameters
        ----------
        snapshot_number : int
            Number of the snapshot for which the entire test inputs.

        Returns
        -------
        torch.Tensor
            Tensor holding the gradient.

        """
        if self.parameters.use_lazy_loading:
            # This fails if an incorrect snapshot was loaded.
            if self.test_data_set.currently_loaded_file != snapshot_number:
                raise Exception("Cannot calculate gradients, wrong file "
                                "was lazily loaded.")
            return self.test_data_set.input_data.grad
        else:
            return self.test_data_inputs.\
                       grad[self.grid_size*snapshot_number:
                            self.grid_size*(snapshot_number+1)]

    def __check_snapshots(self):
        """Check the snapshots for consistency."""
        self.nr_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots using a memorymap to see if there is consistency.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            printout("Checking descriptor file ", snapshot.input_npy_file,
                     "at", snapshot.input_npy_directory, min_verbosity=1)
            tmp = self.__load_from_npy_file(
                os.path.join(snapshot.input_npy_directory,
                             snapshot.input_npy_file), mmapmode='r')

            # We have to cut xyz information, if we have xyz information in
            # the descriptors.
            if self.descriptor_calculator.descriptors_contain_xyz:
                # Remove first 3 elements of descriptors, as they correspond
                # to the x,y and z information.
                tmp = tmp[:, :, :, 3:]

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_input_dimension = np.shape(tmp)[-1]
            tmp_grid_dim = np.shape(tmp)[0:3]
            if firstsnapshot:
                self.input_dimension = tmp_input_dimension
                self.grid_dimension[0:3] = tmp_grid_dim[0:3]
            else:
                if (self.input_dimension != tmp_input_dimension
                        or self.grid_dimension[0] != tmp_grid_dim[0]
                        or self.grid_dimension[1] != tmp_grid_dim[1]
                        or self.grid_dimension[2] != tmp_grid_dim[2]):
                    raise Exception("Invalid snapshot entered at ", snapshot.
                                    input_npy_file)

            ####################
            # Targets.
            ####################

            printout("Checking targets file ", snapshot.output_npy_file, "at",
                     snapshot.output_npy_directory, min_verbosity=1)
            tmp_out = self.__load_from_npy_file(
                os.path.join(snapshot.output_npy_directory,
                             snapshot.output_npy_file), mmapmode='r')

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = np.shape(tmp_out)[-1]
            tmp_grid_dim = np.shape(tmp_out)[0:3]
            if firstsnapshot:
                self.output_dimension = tmp_output_dimension
            else:
                if self.output_dimension != tmp_output_dimension:
                    raise Exception("Invalid snapshot entered at ", snapshot.
                                    output_npy_file)
            if (self.grid_dimension[0] != tmp_grid_dim[0]
                    or self.grid_dimension[1] != tmp_grid_dim[1]
                    or self.grid_dimension[2] != tmp_grid_dim[2]):
                raise Exception("Invalid snapshot entered at ", snapshot.
                                output_npy_file)

            if firstsnapshot:
                firstsnapshot = False

        # Save the grid size.
        self.grid_size = self.grid_dimension[0]\
            * self.grid_dimension[1] * self.grid_dimension[2]

        # Now we need to confirm that the snapshot list has some inner
        # consistency.
        if self.parameters.data_splitting_type == "by_snapshot":
            snapshot: Snapshot
            for snapshot in self.parameters.snapshot_directories_list:
                if snapshot.snapshot_function == "tr":
                    self.nr_training_snapshots += 1
                elif snapshot.snapshot_function == "te":
                    self.nr_test_snapshots += 1
                elif snapshot.snapshot_function == "va":
                    self.nr_validation_snapshots += 1
                else:
                    raise Exception("Unknown option for snapshot splitting "
                                    "selected.")

            # Now we need to check whether or not this input is believable.
            nr_of_snapshots = len(self.parameters.snapshot_directories_list)
            if nr_of_snapshots != (self.nr_training_snapshots +
                                   self.nr_test_snapshots +
                                   self.nr_validation_snapshots):
                raise Exception("Cannot split snapshots with specified "
                                "splitting scheme, "
                                "too few or too many options selected")
            if self.nr_training_snapshots == 0 and self.nr_test_snapshots == 0:
                raise Exception("No training snapshots provided.")
            if self.nr_validation_snapshots == 0 and \
                    self.nr_test_snapshots == 0:
                raise Exception("No validation snapshots provided.")
            if self.nr_training_snapshots == 0 and self.nr_test_snapshots != 0:
                printout("DataHandler prepared for inference. No training "
                         "possible with this setup. "
                         "If this is not what you wanted, please revise the "
                         "input script.", min_verbosity=0)
                if self.nr_validation_snapshots != 0:
                    printout("As this DataHandler can only be used for "
                             "inference, the validation data you have "
                             "provided will be ignored.", min_verbosity=1)
            if self.nr_test_snapshots == 0:
                printout("Running MALA without test data. If this is not "
                         "what you wanted, "
                         "please revise the input script.", min_verbosity=0)

        else:
            raise Exception("Wrong parameter for data splitting provided.")

        # As we are not actually interested in the number of snapshots, but in
        # the number of datasets,we need to multiply by that.
        self.nr_training_data = self.nr_training_snapshots*self.grid_size
        self.nr_validation_data = self.nr_validation_snapshots*self.grid_size
        self.nr_test_data = self.nr_test_snapshots*self.grid_size

        # Reordering the lists.
        snapshot_order = {'tr': 0, 'va': 1, 'te': 2}
        self.parameters.snapshot_directories_list.sort(key=lambda d:
                                                       snapshot_order
                                                       [d.snapshot_function])

    def __load_from_npy_file(self, file, mmapmode=None):
        """Load a numpy array from a file."""
        loaded_array = np.load(file, mmap_mode=mmapmode)
        if len(self.dbg_grid_dimensions) == 3:
            return loaded_array[0:self.dbg_grid_dimensions[0],
                                0:self.dbg_grid_dimensions[1],
                                0:self.dbg_grid_dimensions[2], :]

        else:
            return loaded_array

    def __parametrize_scalers(self):
        """Use the training data to parametrize the DataScalers."""
        ##################
        # Inputs.
        ##################

        # If we do lazy loading, we have to iterate over the files one at a
        # time and add them to the fit, i.e. incrementally updating max/min
        # or mean/std. If we DON'T do lazy loading, we can simply load the
        # training data (we will need it later anyway) and perform the
        # scaling. This should save some performance.

        if self.parameters.use_lazy_loading:
            self.input_data_scaler.start_incremental_fitting()
            # We need to perform the data scaling over the entirety of the
            # training data.
            for snapshot in self.parameters.snapshot_directories_list:
                # Data scaling is only performed on the training data sets.
                if snapshot.snapshot_function == "tr":
                    tmp = self.__load_from_npy_file(os.path.join(snapshot.
                                                    input_npy_directory,
                                                    snapshot.input_npy_file),
                                                    mmapmode='r')
                    if self.descriptor_calculator.descriptors_contain_xyz:
                        tmp = tmp[:, :, :, 3:]

                    # The scalers will later operate on torch Tensors so we
                    # have to make sure they are fitted on
                    # torch Tensors as well. Preprocessing the numpy data as
                    # follows does NOT load it into memory, see
                    # test/tensor_memory.py
                    tmp = np.array(tmp)
                    tmp *= self.descriptor_calculator.\
                        convert_units(1, snapshot.input_units)
                    tmp = tmp.astype(np.float32)
                    tmp = tmp.reshape([self.grid_size,
                                       self.get_input_dimension()])
                    tmp = torch.from_numpy(tmp).float()
                    self.input_data_scaler.incremental_fit(tmp)

            self.input_data_scaler.finish_incremental_fitting()

        else:
            self.__load_training_data_into_ram()
            self.input_data_scaler.fit(self.training_data_inputs)

        printout("Input scaler parametrized.", min_verbosity=1)

        ##################
        # Output.
        ##################

        # If we do lazy loading, we have to iterate over the files one at a
        # time and add them to the fit,
        # i.e. incrementally updating max/min or mean/std.
        # If we DON'T do lazy loading, we can simply load the training data
        # (we will need it later anyway)
        # and perform the scaling. This should save some performance.

        if self.parameters.use_lazy_loading:
            i = 0
            self.output_data_scaler.start_incremental_fitting()
            # We need to perform the data scaling over the entirety of the
            # training data.
            for snapshot in self.parameters.snapshot_directories_list:
                # Data scaling is only performed on the training data sets.
                if snapshot.snapshot_function == "tr":
                    tmp = self.__load_from_npy_file(os.path.join(snapshot.
                                                    output_npy_directory,
                                                    snapshot.output_npy_file),
                                                    mmapmode='r')
                    # The scalers will later operate on torch Tensors so we
                    # have to make sure they are fitted on
                    # torch Tensors as well. Preprocessing the numpy data as
                    # follows does NOT load it into memory, see
                    # test/tensor_memory.py
                    tmp = np.array(tmp)
                    tmp *= self.target_calculator.\
                        convert_units(1, snapshot.output_units)
                    tmp = tmp.astype(np.float32)
                    tmp = tmp.reshape([self.grid_size,
                                       self.get_output_dimension()])
                    tmp = torch.from_numpy(tmp).float()
                    self.output_data_scaler.incremental_fit(tmp)
                i += 1
            self.output_data_scaler.finish_incremental_fitting()

        else:
            # Already loaded into RAM above.
            self.output_data_scaler.fit(self.training_data_outputs)

        printout("Output scaler parametrized.", min_verbosity=1)

    def __load_training_data_into_ram(self):
        """Load the training data into RAM."""
        # INPUTS.

        self.training_data_inputs = []
        # We need to perform the data scaling over the entirety of
        # the training data.
        for snapshot in self.parameters.snapshot_directories_list:

            # Data scaling is only performed on the training data sets.
            if snapshot.snapshot_function == "tr":
                tmp = self.__load_from_npy_file(os.path.join(snapshot.
                                                input_npy_directory,
                                                snapshot.input_npy_file),
                                                mmapmode='r')
                if self.descriptor_calculator.descriptors_contain_xyz:
                    tmp = tmp[:, :, :, 3:]
                tmp = np.array(tmp)
                tmp *= self.descriptor_calculator. \
                    convert_units(1, snapshot.input_units)
                self.training_data_inputs.append(tmp)

        # The scalers will later operate on torch Tensors so we have to
        # make sure they are fitted on
        # torch Tensors as well. Preprocessing the numpy data as follows
        # does NOT load it into memory, see
        # test/tensor_memory.py
        self.training_data_inputs = np.array(self.training_data_inputs)
        self.training_data_inputs = \
            self.training_data_inputs.astype(np.float32)
        self.training_data_inputs = \
            self.training_data_inputs.reshape(
                [self.nr_training_data, self.get_input_dimension()])
        self.training_data_inputs = \
            torch.from_numpy(self.training_data_inputs).float()

        # Outputs.
        self.training_data_outputs = []
        # We need to perform the data scaling over the entirety of
        # the training data.
        for snapshot in self.parameters.snapshot_directories_list:

            # Data scaling is only performed on the training data sets.
            if snapshot.snapshot_function == "tr":
                tmp = self. \
                    __load_from_npy_file(os.path.join(
                                         snapshot.output_npy_directory,
                                         snapshot.output_npy_file),
                                         mmapmode='r')
                tmp = np.array(tmp)
                tmp *= self.target_calculator. \
                    convert_units(1, snapshot.output_units)
                self.training_data_outputs.append(tmp)

        # The scalers will later operate on torch Tensors so we have to
        # make sure they are fitted on
        # torch Tensors as well. Preprocessing the numpy data as follows
        # does NOT load it into memory, see
        # test/tensor_memory.py
        self.training_data_outputs = np.array(self.training_data_outputs)
        self.training_data_outputs = \
            self.training_data_outputs.astype(np.float32)
        self.training_data_outputs = self.training_data_outputs.reshape(
            [self.nr_training_data, self.get_output_dimension()])
        self.training_data_outputs = \
            torch.from_numpy(self.training_data_outputs).float()

    def __build_datasets(self):
        """Build the DataSets that are used during training."""
        if self.parameters.use_lazy_loading:

            # Create the lazy loading data sets.
            if self.parameters.use_clustering:
                self.training_data_set = LazyLoadDatasetClustered(
                    self.get_input_dimension(), self.get_output_dimension(),
                    self.input_data_scaler, self.output_data_scaler,
                    self.descriptor_calculator, self.target_calculator,
                    self.grid_dimension, self.grid_size,
                    self.use_horovod, self.parameters.number_of_clusters,
                    self.parameters.train_ratio,
                    self.parameters.sample_ratio)
                self.validation_data_set = LazyLoadDataset(
                    self.get_input_dimension(), self.get_output_dimension(),
                    self.input_data_scaler, self.output_data_scaler,
                    self.descriptor_calculator, self.target_calculator,
                    self.grid_dimension, self.grid_size,
                    self.use_horovod)

                if self.nr_test_data != 0:
                    self.test_data_set = LazyLoadDataset(
                        self.get_input_dimension(),
                        self.get_output_dimension(),
                        self.input_data_scaler, self.output_data_scaler,
                        self.descriptor_calculator, self.target_calculator,
                        self.grid_dimension, self.grid_size,
                        self.use_horovod,
                        input_requires_grad=True)

            else:
                self.training_data_set = LazyLoadDataset(
                    self.get_input_dimension(), self.get_output_dimension(),
                    self.input_data_scaler, self.output_data_scaler,
                    self.descriptor_calculator, self.target_calculator,
                    self.grid_dimension, self.grid_size,
                    self.use_horovod)
                self.validation_data_set = LazyLoadDataset(
                    self.get_input_dimension(), self.get_output_dimension(),
                    self.input_data_scaler, self.output_data_scaler,
                    self.descriptor_calculator, self.target_calculator,
                    self.grid_dimension, self.grid_size,
                    self.use_horovod)

                if self.nr_test_data != 0:
                    self.test_data_set = LazyLoadDataset(
                        self.get_input_dimension(),
                        self.get_output_dimension(),
                        self.input_data_scaler, self.output_data_scaler,
                        self.descriptor_calculator, self.target_calculator,
                        self.grid_dimension, self.grid_size,
                        self.use_horovod,
                        input_requires_grad=True)

            # Add snapshots to the lazy loading data sets.
            for snapshot in self.parameters.snapshot_directories_list:
                if snapshot.snapshot_function == "tr":
                    self.training_data_set.add_snapshot_to_dataset(snapshot)
                if snapshot.snapshot_function == "va":
                    self.validation_data_set.add_snapshot_to_dataset(snapshot)
                if snapshot.snapshot_function == "te":
                    self.test_data_set.add_snapshot_to_dataset(snapshot)

            if self.parameters.use_clustering:
                self.training_data_set.cluster_dataset()
            # I don't think we need to mix them here. We can use the standard
            # ordering for the first epoch
            # and mix it up after.
            # self.training_data_set.mix_datasets()
            # self.validation_data_set.mix_datasets()
            # self.test_data_set.mix_datasets()
        else:
            # We iterate through the snapshots and add the validation data and
            # test data.
            self.validation_data_inputs = []
            self.validation_data_outputs = []
            if self.nr_test_data != 0:
                self.test_data_inputs = []
                self.test_data_outputs = []

            # We need to perform the data scaling over the entirety of the
            # training data.
            for snapshot in self.parameters.snapshot_directories_list:

                # Data scaling is only performed on the training data sets.
                if snapshot.snapshot_function == "va" \
                        or snapshot.snapshot_function == "te":
                    tmp = self.__load_from_npy_file(
                        os.path.join(snapshot.input_npy_directory,
                                     snapshot.input_npy_file), mmapmode='r')
                    if self.descriptor_calculator.descriptors_contain_xyz:
                        tmp = tmp[:, :, :, 3:]
                    tmp = np.array(tmp)
                    tmp *= self.descriptor_calculator.\
                        convert_units(1, snapshot.input_units)
                    if snapshot.snapshot_function == "va":
                        self.validation_data_inputs.append(tmp)
                    if snapshot.snapshot_function == "te":
                        self.test_data_inputs.append(tmp)
                    tmp = self.__load_from_npy_file(
                        os.path.join(snapshot.output_npy_directory,
                                     snapshot.output_npy_file), mmapmode='r')
                    tmp = np.array(tmp)
                    tmp *= self.target_calculator.\
                        convert_units(1, snapshot.output_units)
                    if snapshot.snapshot_function == "va":
                        self.validation_data_outputs.append(tmp)
                    if snapshot.snapshot_function == "te":
                        self.test_data_outputs.append(tmp)

            # I know this would be more elegant with the member functions typed
            # below. But I am pretty sure
            # that that would create a temporary copy of the arrays, and that
            # could overload the RAM
            # in cases where lazy loading is technically not even needed.
            if self.nr_test_data != 0:
                self.test_data_inputs = np.array(self.test_data_inputs)
                self.test_data_inputs = \
                    self.test_data_inputs.astype(np.float32)
                self.test_data_inputs = \
                    self.test_data_inputs.reshape(
                        [self.nr_test_data, self.get_input_dimension()])
                self.test_data_inputs = \
                    torch.from_numpy(self.test_data_inputs).float()
                self.test_data_inputs = \
                    self.input_data_scaler.transform(self.test_data_inputs)
                self.test_data_inputs.requires_grad = True

            self.validation_data_inputs = np.array(self.validation_data_inputs)
            self.validation_data_inputs = \
                self.validation_data_inputs.astype(np.float32)
            self.validation_data_inputs = \
                self.validation_data_inputs.reshape(
                    [self.nr_validation_data, self.get_input_dimension()])
            self.validation_data_inputs = \
                torch.from_numpy(self.validation_data_inputs).float()
            self.validation_data_inputs = \
                self.input_data_scaler.transform(self.validation_data_inputs)
            self.training_data_inputs = \
                self.input_data_scaler.transform(self.training_data_inputs)

            if self.nr_test_data != 0:
                self.test_data_outputs = np.array(self.test_data_outputs)
                self.test_data_outputs = \
                    self.test_data_outputs.astype(np.float32)
                self.test_data_outputs = \
                    self.test_data_outputs.reshape(
                        [self.nr_test_data, self.get_output_dimension()])
                self.test_data_outputs = \
                    torch.from_numpy(self.test_data_outputs).float()
                self.test_data_outputs = \
                    self.output_data_scaler.transform(self.test_data_outputs)

            self.validation_data_outputs = \
                np.array(self.validation_data_outputs)
            self.validation_data_outputs = \
                self.validation_data_outputs.astype(np.float32)
            self.validation_data_outputs = \
                self.validation_data_outputs.reshape(
                    [self.nr_validation_data, self.get_output_dimension()])
            self.validation_data_outputs = \
                torch.from_numpy(self.validation_data_outputs).float()
            self.validation_data_outputs = \
                self.output_data_scaler.transform(self.validation_data_outputs)
            self.training_data_outputs = \
                self.output_data_scaler.transform(self.training_data_outputs)

            if self.nr_training_data != 0:
                self.training_data_set = \
                    TensorDataset(self.training_data_inputs,
                                  self.training_data_outputs)
            if self.nr_validation_data != 0:
                self.validation_data_set = \
                    TensorDataset(self.validation_data_inputs,
                                  self.validation_data_outputs)
            if self.nr_test_data != 0:
                self.test_data_set = \
                    TensorDataset(self.test_data_inputs,
                                  self.test_data_outputs)

    def __raw_numpy_to_converted_numpy(self, numpy_array, data_type="in",
                                       units=None):
        """Convert a raw numpy array containing into the correct units."""
        if data_type == "in":
            if data_type == "in" and self.descriptor_calculator.\
                    descriptors_contain_xyz:
                numpy_array = numpy_array[:, :, :, 3:]
            if units is not None:
                numpy_array *= self.descriptor_calculator.convert_units(1,
                                                                        units)
            return numpy_array
        elif data_type == "out":
            if units is not None:
                numpy_array *= self.target_calculator.convert_units(1, units)
            return numpy_array
        else:
            raise Exception("Please choose either \"in\" or \"out\" for "
                            "this function.")

    def __converted_numpy_to_scaled_tensor(self, numpy_array,
                                           desired_dimensions=None,
                                           data_type="in"):
        """
        Transform a numpy array containing into a scaled torch tensor.

        This tensor that can simply be put into a MALA network.
        No unit conversion is done here.
        """
        numpy_array = numpy_array.astype(np.float32)
        if desired_dimensions is not None:
            numpy_array = numpy_array.reshape(desired_dimensions)
        numpy_array = torch.from_numpy(numpy_array).float()
        if data_type == "in":
            numpy_array = self.input_data_scaler.transform(numpy_array)
        elif data_type == "out":
            numpy_array = self.output_data_scaler.transform(numpy_array)
        else:
            raise Exception("Please choose either \"in\" or \"out\" for "
                            "this function.")
        return numpy_array
