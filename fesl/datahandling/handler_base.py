import numpy as np
import torch
from torch.utils.data import TensorDataset


class HandlerBase:
    """Base class for all data objects."""

    def __init__(self, p, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser):
        self.parameters = p.data
        self.raw_input_grid = []
        """
        Input data (descriptors) in the form of 
        
        number_of_snapshots x gridx x gridy x gridz x feature_length.
        
        This array might be empty - depending on the form of input parser that is used.
        It will always have the datatype np.float32.
        If needed, unit conversion is done on this data.
        """
        self.raw_output_grid = []
        """
        Output data (targets) in the form of 

        number_of_snapshots x gridx x gridy x gridz x feature_length.

        This array might be empty - depending on the form of output parser that is used.
        It will always have the datatype np.float32.
        If needed, unit conversion is done on this data.
        """
        self.raw_input_datasize = np.zeros(1, dtype=np.float32)
        """
        Input data (descriptors) in the form of 

        number_of_snapshots x (gridx x gridy x gridz) x feature_length.

        This array will always have data after load_data, either because the data was directly read into it or 
        because it was reshaped.
        It will always have the datatype np.float32.
        If needed, unit conversion is done on this data.
        """
        self.raw_output_datasize = np.zeros(1, dtype=np.float32)
        """
        Output data (targets) in the form of 

        number_of_snapshots x (gridx x gridy x gridz) x feature_length.

        This array will always have data after load_data, either because the data was directly read into it or 
        because it was reshaped.
        It will always have the datatype np.float32.
        If needed, unit conversion is done on this data.
        """

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

        self.inference_data_inputs = torch.empty(0)
        """
        Torch tensor holding all scaled inference data inputs.
        """

        self.inference_data_outputs = torch.empty(0)
        """
        Torch tensor holding all scaled inference data outputs.
        """

        self.training_data_set = torch.empty(0)
        """
        Torch dataset holding all scaled training data.
        """

        self.validation_data_set = torch.empty(0)
        """
        Torch dataset holding all scaled validation data.
        """

        self.test_data_set = torch.empty(0)
        """
        Torch dataset holding all scaled testing data.
        """

        self.nr_snapshots = 0
        """
        Number of screenshots in this datahandler.
        """

        self.nr_training_data = 0
        """
        Number of training data points.
        """
        self.nr_test_data = 0
        """
        Number of testing data points.
        """

        self.nr_validation_data = 0
        """
        Number of validation data points.
        """

        self.grid_dimension = np.array([0, 0, 0])
        """
        Dimension of the real space grid uses in the underlying calculation, as list [x,y,z].
        """

        self.grid_size = 0
        """
        Dimension of the real space grid uses in the underlying calculation, as integer x*y*z.
        """

        self.input_data_scaler = input_data_scaler
        """
        Scaler object that can be used to scale input data. 
        Scaling DOES NOT include unit conversion. Descriptor objects are used for that.
        """

        self.output_data_scaler = output_data_scaler
        """
        Scaler object that can be used to scale output data. 
        Scaling DOES NOT include unit conversion. Target objects are used for that.
        """

        self.target_parser = target_parser
        """
        Parses targets (physical quantities such as LDOS) and handles unit conversion.
        """

        self.descriptor_calculator = descriptor_calculator
        """
        Parses descriptors (e.g. SNAP) and handles unit conversion.
        """


    def load_data(self):
        """Loads data from the specified directory.
        The way this is done is provided by the base classes. At the end of load_data raw_input_datasize
        and raw_output_datasize exist with the dimensions

        number_of_snapshots x (gridx x gridy x gridz) x feature_length

        and in the correct units.
        """
        raise Exception("load_data not implemented.")

    def prepare_data(self):
        """Prepares the data to be used in an ML workflow TRAINING.
        This includes the following operations:
        1. Data splitting (in training, validation and testing).
        2. Data scaling
        3. Building of tensor data sets.
        """

        # NOTE: We nornmalize the data AFTER we split it because the splitting might be based on
        # snapshot "borders", i.e. training, validation and test set will be composed of an
        # integer number of snapshots. Therefore we cannot convert the data into PyTorch
        # tensors of dimension (number_of_points,feature_length) until AFTER we have
        # split it, and normalization is more convenient after this conversion.

        # Split from raw numpy arrays into three/six pytorch tensors.
        self.split_data()

        # Normalize pytorch tensors.
        self.scale_data()

        # Build dataset objects from pytorch tensors.
        self.build_datasets()

    def prepare_data_for_inference(self):
        """
        Prepares the data to be used in an ML workflow inference.
        This includes the following operations:
        1. Transformation to pytorch tensor.
        2. Data scaling

        It is always done by snapshot.
        """

        # Transform data to pytorch tensor.
        self.inference_data_inputs = torch.from_numpy(self.raw_input_datasize).float()
        self.inference_data_outputs = torch.from_numpy(self.raw_output_datasize).float()

        # Scale the data.
        self.inference_data_inputs = self.input_data_scaler.transform(self.inference_data_inputs)
        self.inference_data_outputs = self.output_data_scaler.transform(self.inference_data_outputs)



    def save_loaded_data(self, directory=".", naming_scheme_input="snapshot_*_1.in",
                         naming_scheme_output="snapshot_*_1.out", filetype="*.npy", raw_data_kind="grid"):
        """Saves the loaded snapshots. Note that saving "datasize" type data might be in converted units,
        if this function is called after prepare_data()."""

        if raw_data_kind != "grid" and raw_data_kind != "datasize":
            raise Exception("Invalid raw data kind requested, could not save raw data.")

        # Inputs.
        print("Saving ", self.nr_snapshots, " snapshot inputs at", directory)
        for i in range(0, self.nr_snapshots):
            if filetype == "*.npy":
                input_units = self.parameters.snapshot_directories_list[i].input_units
                save_path = directory+naming_scheme_input.replace("*", str(i))
                if raw_data_kind == "grid":
                    np.save(save_path, self.descriptor_calculator.backconvert_units(self.raw_input_grid[i], input_units), allow_pickle=True)
                if raw_data_kind == "datasize":
                    np.save(save_path, self.descriptor_calculator.backconvert_units(self.raw_input_datasize[i], input_units), allow_pickle=True)

        # Outputs.
        print("Saving ", self.nr_snapshots, " snapshot outputs at", directory)
        for i in range(0, self.nr_snapshots):
            if filetype == "*.npy":
                output_units = self.parameters.snapshot_directories_list[i].output_units
                save_path = directory+naming_scheme_output.replace("*", str(i))
                if raw_data_kind == "grid":
                    np.save(save_path, self.target_parser.backconvert_units(self.raw_output_grid[i], output_units), allow_pickle=True)
                if raw_data_kind == "datasize":
                    np.save(save_path, self.target_parser.backconvert_units(self.raw_output_datasize[i], output_units), allow_pickle=True)

    def save_prepared_data(self):
        """Saves the data after it was altered/preprocessed by the ML workflow. E.g. SNAP descriptor calculation."""
        raise Exception("save_data not implemented.")

    def get_input_dimension(self):
        """Returns the dimension of the input vector."""
        raise Exception("get_input_dimension not implemented.")

    def get_output_dimension(self):
        """Returns the dimension of the output vector."""
        raise Exception("get_output_dimension not implemented.")

    def dbg_reduce_number_of_data_points(self, newnumber):
        raise Exception("dbg_reduce_number_of_data_points not implemented.")

    def split_data(self):
        """Splits data into training, validation and test data sets.
        Performs a snapshot->pytorch tensor conversion, if necessary.
        """
        if self.parameters.data_splitting_type == "random":
            self.split_data_randomly()
        elif self.parameters.data_splitting_type == "by_snapshot":
            self.split_data_by_snapshot()
        else:
            raise Exception("Wrong parameter for data splitting provided.")

    def split_data_randomly(self):
        """This function splits the data randomly, i.e. specified portions
        of the data are used as test, validation and training data. This can lead
        to errors, since we do not enforce equal representation of certain clusters."""

        tmp_in = self.raw_input_datasize.reshape([self.grid_size*self.nr_snapshots, self.get_input_dimension()])
        tmp_out = self.raw_output_datasize.reshape(self.grid_size*self.nr_snapshots, self.get_output_dimension())

        if sum(self.parameters.data_splitting_percent) != 100:
            raise Exception("Could not split data randomly - will not attempt to use anything but "
                            "100% of provided data.")

        # Split data according to parameters.
        # raw_input and raw_ouput are guaranteed to provide the same first dimensions
        # as they are calculated using the grid and the number of snapshots.
        # We enforce that the grid size is equal in load_data().
        self.nr_training_data = int(self.parameters.data_splitting_percent[0] / 100 * np.shape(tmp_in)[0])
        self.nr_validation_data = int(self.parameters.data_splitting_percent[1] / 100 * np.shape(tmp_in)[0])
        self.nr_test_data = int(self.parameters.data_splitting_percent[2] / 100 * np.shape(tmp_in)[0])

        # We need to make sure that really all of the data is used.
        missing_data = (self.nr_training_data + self.nr_validation_data + self.nr_test_data) - np.shape(tmp_in)[0]
        self.nr_test_data += missing_data

        # Determine the indices at which to split.
        index1 = self.nr_training_data
        index2 = self.nr_training_data + self.nr_validation_data

        # Split the data into three sets, create a tensor for input and output.
        self.training_data_inputs = torch.from_numpy(tmp_in[0:index1]).float()
        self.validation_data_inputs = torch.from_numpy(tmp_in[index1:index2]).float()
        self.test_data_inputs = torch.from_numpy(tmp_in[index2:]).float()
        self.training_data_outputs = torch.from_numpy(tmp_out[0:index1]).float()
        self.validation_data_outputs = torch.from_numpy(tmp_out[index1:index2]).float()
        self.test_data_outputs = torch.from_numpy(tmp_out[index2:]).float()

    def split_data_by_snapshot(self):
        """This function splits the data by snapshot i.e. a certain number of snapshots is training,
        validation and testing."""

        # First, we need to find out how many snapshots we have for training, validation and testing.
        self.nr_training_data = 0
        self.nr_validation_data = 0
        self.nr_test_data = 0

        for snapshot_function in self.parameters.data_splitting_snapshots:
            if snapshot_function == "tr":
                self.nr_training_data += 1
            elif snapshot_function == "te":
                self.nr_test_data += 1
            elif snapshot_function == "va":
                self.nr_validation_data += 1
            else:
                raise Exception("Unknown option for snapshot splitting selected.")

        # Now we need to check whether or not this input is believable.
        nr_of_snapshots = len(self.parameters.snapshot_directories_list)
        if nr_of_snapshots != (self.nr_training_data+self.nr_validation_data+self.nr_test_data):
            raise Exception("Cannot split snapshots with specified splitting scheme, "
                            "too few or too many options selected")
        if self.nr_training_data == 0:
            raise Exception("No training snapshots provided.")
        if self.nr_validation_data == 0:
            raise Exception("No validation snapshots provided.")
        if self.nr_test_data == 0:
            raise Exception("No testing snapshots provided.")

        # Prepare temporary arrays for training, test and validation data

        tmp_training_in = np.zeros((self.nr_training_data, self.grid_size, self.get_input_dimension()), dtype=np.float32)
        tmp_validation_in = np.zeros((self.nr_validation_data, self.grid_size, self.get_input_dimension()), dtype=np.float32)
        tmp_test_in = np.zeros((self.nr_test_data, self.grid_size, self.get_input_dimension()), dtype=np.float32)

        tmp_training_out = np.zeros((self.nr_training_data, self.grid_size, self.get_output_dimension()), dtype=np.float32)
        tmp_validation_out = np.zeros((self.nr_validation_data, self.grid_size, self.get_output_dimension()), dtype=np.float32)
        tmp_test_out = np.zeros((self.nr_test_data, self.grid_size, self.get_output_dimension()), dtype=np.float32)

        # Iterate through list that commands the splitting.
        counter_training = 0
        counter_validation = 0
        counter_testing = 0
        for i in range(0, nr_of_snapshots):
            snapshot_function = self.parameters.data_splitting_snapshots[i]
            if snapshot_function == "tr":
                tmp_training_in[counter_training, :, :] = self.raw_input_datasize[i, :, :]
                tmp_training_out[counter_training, :, :] = self.raw_output_datasize[i, :, :]
                counter_training += 1
            if snapshot_function == "va":
                tmp_validation_in[counter_validation, :, :] = self.raw_input_datasize[i, :, :]
                tmp_validation_out[counter_validation, :, :] = self.raw_output_datasize[i, :, :]
                counter_validation += 1
            if snapshot_function == "te":
                tmp_test_in[counter_testing, :, :] = self.raw_input_datasize[i, :, :]
                tmp_test_out[counter_testing, :, :] = self.raw_output_datasize[i, :, :]
                counter_testing += 1

        # Get the actual data count.
        gridsize = self.grid_dimension[0] * self.grid_dimension[1] *  self.grid_dimension[2]
        self.nr_training_data = gridsize * self.nr_training_data
        self.nr_validation_data = gridsize * self.nr_validation_data
        self.nr_test_data = gridsize * self.nr_test_data

        # Reshape the temporary arrays.
        tmp_training_in = tmp_training_in.reshape([self.nr_training_data, self.get_input_dimension()])
        tmp_validation_in = tmp_validation_in.reshape([self.nr_validation_data, self.get_input_dimension()])
        tmp_test_in = tmp_test_in.reshape([self.nr_test_data, self.get_input_dimension()])

        tmp_training_out = tmp_training_out.reshape([self.nr_training_data, self.get_output_dimension()])
        tmp_validation_out = tmp_validation_out.reshape([self.nr_validation_data, self.get_output_dimension()])
        tmp_test_out = tmp_test_out.reshape([self.nr_test_data, self.get_output_dimension()])

        # Create the tensors.
        self.training_data_inputs = torch.from_numpy(tmp_training_in).float()
        self.validation_data_inputs = torch.from_numpy(tmp_validation_in).float()
        self.test_data_inputs = torch.from_numpy(tmp_test_in).float()
        self.training_data_outputs = torch.from_numpy(tmp_training_out).float()
        self.validation_data_outputs = torch.from_numpy(tmp_validation_out).float()
        self.test_data_outputs = torch.from_numpy(tmp_test_out).float()

    def build_datasets(self):
        """Takes the normalized training, test and validation data and builds data sets with them."""
        self.training_data_set = TensorDataset(self.training_data_inputs, self.training_data_outputs)
        self.validation_data_set = TensorDataset(self.validation_data_inputs, self.validation_data_outputs)
        self.test_data_set = TensorDataset(self.test_data_inputs, self.test_data_outputs)

    def scale_data(self):
        """Scales the data."""

        # First, prepare the data scalers.
        self.input_data_scaler.fit(self.training_data_inputs)
        self.output_data_scaler.fit(self.training_data_outputs)

        # Inputs.
        self.training_data_inputs = self.input_data_scaler.transform(self.training_data_inputs)
        self.validation_data_inputs = self.input_data_scaler.transform(self.validation_data_inputs)
        self.test_data_inputs = self.input_data_scaler.transform(self.test_data_inputs)

        # Outputs.
        self.training_data_outputs = self.output_data_scaler.transform(self.training_data_outputs)
        self.validation_data_outputs = self.output_data_scaler.transform(self.validation_data_outputs)
        self.test_data_outputs = self.output_data_scaler.transform(self.test_data_outputs)

    def grid_to_datasize(self):
        """Transforms snapshot data from
        number_of_snapshots x gridx x gridy x gridz x feature_length
         to
         (number_of_snapshots x gridx x gridy x gridz) x feature_length.
        """
        self.raw_input_datasize = self.raw_input_grid.reshape([self.nr_snapshots, self.grid_size, self.get_input_dimension()])
        self.raw_output_datasize = self.raw_output_grid.reshape([self.nr_snapshots, self.grid_size, self.get_output_dimension()])
