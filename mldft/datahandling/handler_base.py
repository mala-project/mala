import numpy as np
import torch
from torch.utils.data import TensorDataset


class HandlerBase:
    """Base class for all data objects."""

    def __init__(self, p, input_data_scaler, output_data_scaler):
        self.parameters = p.data
        self.raw_input = []
        self.raw_output = []

        self.training_data_inputs = torch.empty(0)
        self.validation_data_inputs = torch.empty(0)
        self.test_data_inputs = torch.empty(0)

        self.training_data_outputs = torch.empty(0)
        self.validation_data_outputs = torch.empty(0)
        self.test_data_outputs = torch.empty(0)

        self.training_data_set = torch.empty(0)
        self.validation_data_set = torch.empty(0)
        self.test_data_set = torch.empty(0)

        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0
        self.grid_dimension = np.array([0, 0, 0])
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler

    def load_data(self):
        """Loads data from the specified directory."""
        raise Exception("load_data not implemented.")

    def prepare_data(self):
        """Prepares the data to be used in an ML workflow (i.e. splitting, normalization,
        conversion of data structure).
        Please be aware that this function will/might change the shape of raw_input/output, for
        memory reasons. If you intend to use a data handler simply to preprocess and save data,
        do not call this function!"""
        # NOTE: We nornmalize the data AFTER we split it because the splitting might be based on
        # snapshot "borders", i.e. training, validation and test set will be composed of an
        # integer number of snapshots. Therefore we cannot convert the data into PyTorch
        # tensors of dimension (number_of_points,feature_length) until AFTER we have
        # split it, and normalization is more convenient after this conversion.

        # Before doing anything with the data, we make sure it's float32.
        # Elsewise Pytorch will copy, not reference our data.
        self.raw_input = self.raw_input.astype(np.float32)
        self.raw_output = self.raw_output.astype(np.float32)

        # Split from raw numpy arrays into three/six pytorch tensors.
        self.split_data()

        # Normalize pytorch tensors.
        self.scale_data()

        # Build dataset objects from pytorch tensors.
        self.build_datasets()

    def save_loaded_data(self, directory=".", naming_scheme_input="snapshot_*_1.in",
                         naming_scheme_output="snapshot_*_1.out", filetype="*.npy"):
        """Saves the loaded snapshots. This fucntion will behave VERY unexpected after prepare data has been called."""

        # Inputs.
        nr_snapshots_inputs = np.shape(self.raw_input)[0]
        print("Saving ", nr_snapshots_inputs, " snapshot inputs at", directory)
        for i in range(0, nr_snapshots_inputs):
            if filetype == "*.npy":
                save_path = directory+naming_scheme_input.replace("*", str(i))
                np.save(save_path, self.raw_input[i], allow_pickle=True)

        # Outputs.
        nr_snapshots_outputs = np.shape(self.raw_output)[0]
        print("Saving ", nr_snapshots_outputs, " snapshot outputs at", directory)
        for i in range(0, nr_snapshots_outputs):
            if filetype == "*.npy":
                save_path = directory+naming_scheme_output.replace("*", str(i))
                np.save(save_path, self.raw_output[i], allow_pickle=True)

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
            self.raw_to_tensor()
            self.split_data_randomly()
        elif self.parameters.data_splitting_type == "by_snapshot":
            self.split_data_by_snapshot()
        else:
            raise Exception("Wrong parameter for data splitting provided.")

    def split_data_randomly(self):
        """This function splits the data randomly, i.e. specified portions
        of the data are used as test, validation and training data. This can lead
        to errors, since we do not enforce equal representation of certain clusters."""

        np.random.shuffle(self.raw_input)
        np.random.shuffle(self.raw_output)

        if sum(self.parameters.data_splitting_percent) != 100:
            raise Exception("Could not split data randomly - will not attempt to use anything but "
                            "100% of provided data.")

        # Split data according to parameters.
        # raw_input and raw_ouput are guaranteed to have the same first dimensions
        # as they are calculated using the grid and the number of snapshots.
        # We enforce that the grid size is equal in load_data().
        self.nr_training_data = int(self.parameters.data_splitting_percent[0] / 100 * np.shape(self.raw_input)[0])
        self.nr_validation_data = int(self.parameters.data_splitting_percent[1] / 100 * np.shape(self.raw_input)[0])
        self.nr_test_data = int(self.parameters.data_splitting_percent[2] / 100 * np.shape(self.raw_input)[0])

        # We need to make sure that really all of the data is used.
        missing_data = (self.nr_training_data + self.nr_validation_data + self.nr_test_data) - np.shape(self.raw_input)[
            0]
        self.nr_test_data += missing_data

        # Determine the indices at which to split.
        index1 = self.nr_training_data
        index2 = self.nr_training_data + self.nr_validation_data

        # Split the data into three sets, create a tensor for input and output.
        self.training_data_inputs = torch.from_numpy(self.raw_input[0:index1]).float()
        self.validation_data_inputs = torch.from_numpy(self.raw_input[index1:index2]).float()
        self.test_data_inputs = torch.from_numpy(self.raw_input[index2:]).float()
        self.training_data_outputs = torch.from_numpy(self.raw_output[0:index1]).float()
        self.validation_data_outputs = torch.from_numpy(self.raw_output[index1:index2]).float()
        self.test_data_outputs = torch.from_numpy(self.raw_output[index2:]).float()

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

        tmp_training_in = np.zeros((self.nr_training_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_input_dimension()), dtype=np.float32)
        tmp_validation_in = np.zeros((self.nr_validation_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_input_dimension()), dtype=np.float32)
        tmp_test_in = np.zeros((self.nr_test_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_input_dimension()), dtype=np.float32)

        tmp_training_out = np.zeros((self.nr_training_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_output_dimension()), dtype=np.float32)
        tmp_validation_out = np.zeros((self.nr_validation_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_output_dimension()), dtype=np.float32)
        tmp_test_out = np.zeros((self.nr_test_data, self.grid_dimension[0],self.grid_dimension[1],self.grid_dimension[2], self.get_output_dimension()), dtype=np.float32)

        # Iterate through list that commands the splitting.
        counter_training = 0
        counter_validation = 0
        counter_testing = 0
        for i in range(0, nr_of_snapshots):
            snapshot_function = self.parameters.data_splitting_snapshots[i]
            if snapshot_function == "tr":
                tmp_training_in[counter_training, :, :, :, :] = self.raw_input[i, :, :, :, :]
                tmp_training_out[counter_training, :, :, :, :] = self.raw_output[i, :, :, :, :]
                counter_training += 1
            if snapshot_function == "va":
                tmp_validation_in[counter_validation, :, :, :, :] = self.raw_input[i, :, :, :, :]
                tmp_validation_out[counter_validation, :, :, :, :] = self.raw_output[i, :, :, :, :]
                counter_validation += 1
            if snapshot_function == "te":
                tmp_test_in[counter_testing, :, :, :, :] = self.raw_input[i, :, :, :, :]
                tmp_test_out[counter_testing, :, :, :, :] = self.raw_output[i, :, :, :, :]
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

    def raw_to_tensor(self):
        """Transforms snapshot data from
        number_of_snapshots x gridx x gridy x gridz x feature_length
         to
         (number_of_snapshots x gridx x gridy x gridz) x feature_length.
        """
        datacount = \
            self.grid_dimension[0] * self.grid_dimension[1] * \
            self.grid_dimension[2] * len(self.parameters.snapshot_directories_list)
        self.raw_input = self.raw_input.reshape([datacount, self.get_input_dimension()])
        self.raw_output = self.raw_output.reshape([datacount, self.get_output_dimension()])

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
