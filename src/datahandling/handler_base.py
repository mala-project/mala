import numpy as np
import torch
from torch.utils.data import TensorDataset


class HandlerBase:
    """Base class for all data objects."""

    def __init__(self, p, data_scaler):
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
        self.data_scaler = data_scaler

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
            self.snapshot_to_tensor()
            self.split_data_randomly()
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

    def build_datasets(self):
        """Takes the normalized training, test and validation data and builds data sets with them."""
        self.training_data_set = TensorDataset(self.training_data_inputs, self.training_data_outputs)
        self.validation_data_set = TensorDataset(self.validation_data_inputs, self.validation_data_outputs)
        self.test_data_set = TensorDataset(self.test_data_inputs, self.test_data_outputs)

    def snapshot_to_tensor(self):
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

        print(self.training_data_inputs[0])
        print(self.training_data_outputs[0])

        # Inputs.

        self.data_scaler.scale_input_tensor(self.training_data_inputs)
        self.data_scaler.scale_input_tensor(self.validation_data_inputs)
        self.data_scaler.scale_input_tensor(self.test_data_inputs)

        # Outputs.

        self.data_scaler.scale_output_tensor(self.training_data_outputs)
        self.data_scaler.scale_output_tensor(self.validation_data_outputs)
        self.data_scaler.scale_output_tensor(self.test_data_outputs)

        print(self.training_data_inputs[0])
        print(self.training_data_outputs[0])

