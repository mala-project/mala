from .data_scaler import DataScaler
from .snapshot import Snapshot
from fesl.common.parameters import Parameters
from fesl.targets.target_interface import TargetInterface
from fesl.descriptors.descriptor_interface import DescriptorInterface
from fesl.common.printout import printout
import numpy as np
import torch

class DataHandler:
    """
    Handles data. Can only process numpy arrays at the moment. Data that is not in a numpy array can be converted using
    the DataConverter class.
    """

    def __init__(self, p: Parameters, target_calculator=None, descriptor_calculator=None, input_data_scaler=None, output_data_scaler=None):
        self.parameters = p.data
        self.dbg_grid_dimensions = p.debug.grid_dimensions
        self.training_data_set = None

        self.validation_data_set = None

        self.test_data_set = None

        self.input_data_scaler = input_data_scaler
        if self.input_data_scaler is None:
            self.input_data_scaler = DataScaler(self.parameters.input_rescaling_type)

        self.output_data_scaler = output_data_scaler
        if self.output_data_scaler is None:
            self.output_data_scaler = DataScaler(self.parameters.output_rescaling_type)

        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = TargetInterface(p)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = DescriptorInterface(p)

        self.nr_snapshots = 0
        self.grid_dimension = [0,0,0]
        self.grid_size = 0
        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0

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


    def add_snapshot(self, input_npy_file, input_npy_directory, output_npy_file, output_npy_directory,
                     input_units="None", output_units="1/eV"):
        """
        Adds a snapshot to data handler. For this type of data,
        a QuantumEspresso outfile, an outfile from the LDOS calculation and
        a directory containing the cube files
        """
        snapshot = Snapshot(input_npy_file, input_npy_directory, input_units, output_npy_file, output_npy_directory,
                                       output_units)
        self.parameters.snapshot_directories_list.append(snapshot)


    def clear_data(self):
        """
        Resets the entire data handling. Useful when doing multiple investigations in the same python file.

        Returns
        -------
        """
        self.training_data_set = None
        self.validation_data_set = None
        self.test_data_set = None

        self.parameters.snapshot_directories_list = []

    def prepare_data(self, reparametrize_scaler=True):
        """

        Parameters
        ----------
        reparametrize_scaler

        Returns
        -------

        """
        # Do a consistency check of the snapshots so that we don't run into an error later.
        # If there is an error, check_snapshots() will raise an exception.
        printout("Checking the snapshots and your inputs for consistency.")
        self.check_snapshots()
        printout("Consistency check successful.")

        # Parametrize the scalers, if needed.
        printout("Initializing the data scalers.")
        if reparametrize_scaler:
            self.parametrize_scalers()
        printout("Data scalers initialized.")


        # Build Datasets.
        # self.build_datasets()


    def check_snapshots(self):
        """
        Checks whether or not the numpy files in the snapshot directories are consistent.
        Returns
        -------
        """
        self.nr_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots using a memorymap to see if there is consistency.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            printout("Checking descriptor file ", snapshot.input_npy_file, "at", snapshot.input_npy_directory)
            tmp = self.load_from_npy_file(snapshot.input_npy_directory + snapshot.input_npy_file,
                                          mmapmode='r')

            # We have to cut xyz information, if we have xyz information in the descriptors.
            if self.parameters.descriptors_contain_xyz:
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
                    raise Exception("Invalid snapshot entered at ", snapshot.input_npy_file)


            ####################
            # Targets.
            ####################

            printout("Checking targets file ", snapshot.output_npy_file, "at", snapshot.output_npy_directory)
            tmp_out = self.load_from_npy_file(snapshot.output_npy_directory + snapshot.output_npy_file,
                                              mmapmode='r')

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = np.shape(tmp_out)[-1]
            tmp_grid_dim = np.shape(tmp_out)[0:3]
            if firstsnapshot:
                self.output_dimension = tmp_output_dimension
            else:
                if self.output_dimension != tmp_output_dimension:
                    raise Exception("Invalid snapshot entered at ", snapshot.output_npy_file)
            if (self.grid_dimension[0] != tmp_grid_dim[0]
                    or self.grid_dimension[1] != tmp_grid_dim[1]
                    or self.grid_dimension[2] != tmp_grid_dim[2]):
                raise Exception("Invalid snapshot entered at ", snapshot.output_npy_file)

            if firstsnapshot:
                firstsnapshot = False

        # Save the grid size.
        self.grid_size = self.grid_dimension[0] * self.grid_dimension[1] * self.grid_dimension[2]

        # Now we need to confirm that the snapshot list has some inner consistency.
        if self.parameters.data_splitting_type == "by_snapshot":
            for snapshot_function in self.parameters.data_splitting_snapshots:
                if snapshot_function == "tr":
                    self.nr_training_data += 1
                elif snapshot_function == "te":
                    self.nr_test_data += 1
                elif snapshot_function == "va":
                    self.nr_validation_data += 1
                else:
                    raise Exception("Unknown option for snapshot splitting selected.")

            # # Now we need to check whether or not this input is believable.
            # nr_of_snapshots = len(self.parameters.snapshot_directories_list)
            # if nr_of_snapshots != (self.nr_training_data + self.nr_validation_data + self.nr_test_data):
            #     raise Exception("Cannot split snapshots with specified splitting scheme, "
            #                     "too few or too many options selected")
            # if self.nr_training_data == 0:
            #     raise Exception("No training snapshots provided.")
            # if self.nr_validation_data == 0:
            #     raise Exception("No validation snapshots provided.")
            # if self.nr_test_data == 0:
            #     raise Exception("No testing snapshots provided.")

        else:
            raise Exception("Wrong parameter for data splitting provided.")

        # As we are not actually interested in the number of snapshots, but in the number of datasets,
        # we need to multiply by that.
        self.nr_training_data *= self.grid_size
        self.nr_validation_data *= self.grid_size
        self.nr_test_data *= self.grid_size



    def load_from_npy_file(self, file, mmapmode=None):
        loaded_array = np.load(file, mmap_mode=mmapmode)
        if len(self.dbg_grid_dimensions) == 3:
            try:
                return loaded_array[0:self.dbg_grid_dimensions[0], 0:self.dbg_grid_dimensions[1], 0:self.dbg_grid_dimensions[2], :]
            except:
                printout(
                    "Could not use grid reduction, falling back to regular grid. Please check that the debug grid is "
                    "not bigger than the actual grid.")

        else:
            return loaded_array


    def parametrize_scalers(self):

        ##################
        # Inputs.
        ##################

        # If we do lazy loading, we have to iterate over the files one at a time and add them to the fit,
        # i.e. incrementally updating max/min or mean/std.
        # If we DON'T do lazy loading, we can simply load the training data (we will need it later anyway)
        # and perform the scaling. This should save some performance.

        if self.parameters.use_lazy_loading:
            i = 0
            self.input_data_scaler.start_incremental_fitting()
            # We need to perform the data scaling over the entirety of the training data.
            for snapshot in self.parameters.snapshot_directories_list:
                # Data scaling is only performed on the training data sets.
                if self.parameters.data_splitting_snapshots[i] == "tr":
                    tmp = self.load_from_npy_file(snapshot.input_npy_directory + snapshot.input_npy_file,
                                                  mmapmode='r')
                    if self.parameters.descriptors_contain_xyz:
                        tmp = tmp[:, :, :, 3:]

                    # The scalers will later operate on torch Tensors so we have to make sure they are fitted on
                    # torch Tensors as well. Preprocessing the numpy data as follows does NOT load it into memory, see
                    # test/tensor_memory.py
                    tmp = np.array(tmp)
                    tmp *= self.descriptor_calculator.convert_units(1, snapshot.input_units)
                    tmp = tmp.astype(np.float32)
                    tmp = tmp.reshape([self.grid_size, self.get_input_dimension()])
                    tmp = torch.from_numpy(tmp).float()
                    self.input_data_scaler.incremental_fit(tmp)
                    i += 1
            self.input_data_scaler.finish_incremental_fitting()

        else:
            self.training_data_inputs = []
            i = 0
            # We need to perform the data scaling over the entirety of the training data.
            for snapshot in self.parameters.snapshot_directories_list:

                # Data scaling is only performed on the training data sets.
                if self.parameters.data_splitting_snapshots[i] == "tr":
                    tmp = self.load_from_npy_file(snapshot.input_npy_directory + snapshot.input_npy_file,
                                                  mmapmode='r')
                    if self.parameters.descriptors_contain_xyz:
                        tmp = tmp[:, :, :, 3:]
                    tmp = np.array(tmp)
                    tmp *= self.descriptor_calculator.convert_units(1, snapshot.input_units)
                    self.training_data_inputs.append(tmp)
                    i += 1

            # The scalers will later operate on torch Tensors so we have to make sure they are fitted on
            # torch Tensors as well. Preprocessing the numpy data as follows does NOT load it into memory, see
            # test/tensor_memory.py
            self.training_data_inputs = np.array(self.training_data_inputs)
            self.training_data_inputs = self.training_data_inputs.astype(np.float32)
            self.training_data_inputs = self.training_data_inputs.reshape([self.nr_training_data, self.get_input_dimension()])
            self.training_data_inputs = torch.from_numpy(self.training_data_inputs).float()
            self.input_data_scaler.fit(self.training_data_inputs)

        printout("Input scaler parametrized.")
        # ##################
        # # Outputs.
        # ##################

        # If we do lazy loading, we have to iterate over the files one at a time and add them to the fit,
        # i.e. incrementally updating max/min or mean/std.
        # If we DON'T do lazy loading, we can simply load the training data (we will need it later anyway)
        # and perform the scaling. This should save some performance.

        ##################
        # Inputs.
        ##################

        # If we do lazy loading, we have to iterate over the files one at a time and add them to the fit,
        # i.e. incrementally updating max/min or mean/std.
        # If we DON'T do lazy loading, we can simply load the training data (we will need it later anyway)
        # and perform the scaling. This should save some performance.

        if self.parameters.use_lazy_loading:
            i = 0
            self.output_data_scaler.start_incremental_fitting()
            # We need to perform the data scaling over the entirety of the training data.
            for snapshot in self.parameters.snapshot_directories_list:
                # Data scaling is only performed on the training data sets.
                if self.parameters.data_splitting_snapshots[i] == "tr":
                    tmp = self.load_from_npy_file(snapshot.output_npy_directory + snapshot.output_npy_file,
                                                  mmapmode='r')
                    # The scalers will later operate on torch Tensors so we have to make sure they are fitted on
                    # torch Tensors as well. Preprocessing the numpy data as follows does NOT load it into memory, see
                    # test/tensor_memory.py
                    tmp = np.array(tmp)
                    tmp *= self.target_calculator.convert_units(1, snapshot.output_units)
                    tmp = tmp.astype(np.float32)
                    tmp = tmp.reshape([self.grid_size, self.get_output_dimension()])
                    tmp = torch.from_numpy(tmp).float()
                    self.output_data_scaler.incremental_fit(tmp)
                    i += 1
            self.output_data_scaler.finish_incremental_fitting()

        else:
            self.training_data_outputs = []
            i = 0
            # We need to perform the data scaling over the entirety of the training data.
            for snapshot in self.parameters.snapshot_directories_list:

                # Data scaling is only performed on the training data sets.
                if self.parameters.data_splitting_snapshots[i] == "tr":
                    tmp = self.load_from_npy_file(snapshot.output_npy_directory + snapshot.output_npy_file,
                                                  mmapmode='r')
                    tmp = np.array(tmp)
                    tmp *= self.target_calculator.convert_units(1, snapshot.output_units)
                    self.training_data_outputs.append(tmp)
                    i += 1

            # The scalers will later operate on torch Tensors so we have to make sure they are fitted on
            # torch Tensors as well. Preprocessing the numpy data as follows does NOT load it into memory, see
            # test/tensor_memory.py
            self.training_data_outputs = np.array(self.training_data_outputs)
            self.training_data_outputs = self.training_data_outputs.astype(np.float32)
            self.training_data_outputs = self.training_data_outputs.reshape([self.nr_training_data, self.get_output_dimension()])
            self.training_data_outputs = torch.from_numpy(self.training_data_outputs).float()
            self.output_data_scaler.fit(self.training_data_outputs)

        printout("Output scaler parametrized.")



    def get_input_dimension(self):
        """Returns the dimension of the input vector."""
        return self.input_dimension

    def get_output_dimension(self):
        """Returns the dimension of the output vector."""
        return self.output_dimension
