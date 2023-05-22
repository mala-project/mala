"""DataSet for lazy-loading."""
import os

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class.
    pass
import numpy as np
import torch
from torch.utils.data import Dataset

from mala.common.parallelizer import barrier
from mala.common.parameters import DEFAULT_NP_DATA_DTYPE
from mala.datahandling.snapshot import Snapshot


class LazyLoadDataset(torch.utils.data.Dataset):
    """
    DataSet class for lazy loading.

    Only loads snapshots in the memory that are currently being processed.
    Uses a "caching" approach of keeping the last used snapshot in memory,
    until values from a new ones are used. Therefore, shuffling at DataSampler
    / DataLoader level is discouraged to the point that it was disabled.
    Instead, we mix the snapshot load order here ot have some sort of mixing
    at all.

    Parameters
    ----------
    input_dimension : int
        Dimension of an input vector.

    output_dimension : int
        Dimension of an output vector.

    input_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the input data.

    output_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the output data.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data.

    target_calculator : mala.targets.target.Target or derivative
        Used to do unit conversion on output data.

    use_horovod : bool
        If true, it is assumed that horovod is used.

    input_requires_grad : bool
        If True, then the gradient is stored for the inputs.
    """

    def __init__(self, input_dimension, output_dimension, input_data_scaler,
                 output_data_scaler, descriptor_calculator,
                 target_calculator, use_horovod,
                 input_requires_grad=False):
        self.snapshot_list = []
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler
        self.descriptor_calculator = descriptor_calculator
        self.target_calculator = target_calculator
        self.number_of_snapshots = 0
        self.total_size = 0
        self.descriptors_contain_xyz = self.descriptor_calculator.\
            descriptors_contain_xyz
        self.currently_loaded_file = None
        self.input_data = np.empty(0)
        self.output_data = np.empty(0)
        self.use_horovod = use_horovod
        self.return_outputs_directly = False
        self.input_requires_grad = input_requires_grad

    @property
    def return_outputs_directly(self):
        """
        Control whether outputs are actually transformed.

        Has to be False for training. In the testing case,
        Numerical errors are smaller if set to True.
        """
        return self._return_outputs_directly

    @return_outputs_directly.setter
    def return_outputs_directly(self, value):
        self._return_outputs_directly = value

    def add_snapshot_to_dataset(self, snapshot: Snapshot):
        """
        Add a snapshot to a DataSet.

        Afterwards, the DataSet can and will load this snapshot as needed.

        Parameters
        ----------
        snapshot : mala.datahandling.snapshot.Snapshot
            Snapshot that is to be added to this DataSet.

        """
        self.snapshot_list.append(snapshot)
        self.number_of_snapshots += 1
        self.total_size += snapshot.grid_size

    def mix_datasets(self):
        """
        Mix the order of the snapshots.

        With this, there can be some variance between runs.
        """
        used_perm = torch.randperm(self.number_of_snapshots)
        barrier()
        if self.use_horovod:
            used_perm = hvd.broadcast(used_perm, 0)
        self.snapshot_list = [self.snapshot_list[i] for i in used_perm]
        self.get_new_data(0)

    def get_new_data(self, file_index):
        """
        Read a new snapshot into RAM.

        Parameters
        ----------
        file_index : i
            File to be read.
        """
        # Load the data into RAM.
        if self.snapshot_list[file_index].snapshot_type == "numpy":
            self.input_data = self.descriptor_calculator. \
                read_from_numpy_file(
                os.path.join(self.snapshot_list[file_index].input_npy_directory,
                             self.snapshot_list[file_index].input_npy_file),
                                units=self.snapshot_list[file_index].input_units)
            self.output_data = self.target_calculator. \
                read_from_numpy_file(
                os.path.join(self.snapshot_list[file_index].output_npy_directory,
                             self.snapshot_list[file_index].output_npy_file),
                             units=self.snapshot_list[file_index].output_units)

        elif self.snapshot_list[file_index].snapshot_type == "openpmd":
            self.input_data = self.descriptor_calculator. \
                read_from_openpmd_file(
                os.path.join(self.snapshot_list[file_index].input_npy_directory,
                             self.snapshot_list[file_index].input_npy_file))
            self.output_data = self.target_calculator. \
                read_from_openpmd_file(
                os.path.join(self.snapshot_list[file_index].output_npy_directory,
                             self.snapshot_list[file_index].output_npy_file))

        # Transform the data.
        self.input_data = \
            self.input_data.reshape([self.snapshot_list[file_index].grid_size,
                                     self.input_dimension])
        if self.input_data.dtype != DEFAULT_NP_DATA_DTYPE:
            self.input_data = self.input_data.astype(DEFAULT_NP_DATA_DTYPE)
        self.input_data = torch.from_numpy(self.input_data).float()
        self.input_data_scaler.transform(self.input_data)
        self.input_data.requires_grad = self.input_requires_grad

        self.output_data = \
            self.output_data.reshape([self.snapshot_list[file_index].grid_size,
                                      self.output_dimension])
        if self.return_outputs_directly is False:
            self.output_data = np.array(self.output_data)
            if self.output_data.dtype != DEFAULT_NP_DATA_DTYPE:
                self.output_data = self.output_data.astype(DEFAULT_NP_DATA_DTYPE)
            self.output_data = torch.from_numpy(self.output_data).float()
            self.output_data_scaler.transform(self.output_data)

        # Save which data we have currently loaded.
        self.currently_loaded_file = file_index

    def _get_file_index(self, idx, is_slice=False, is_start=False):
        file_index = None
        index_in_file = idx
        if is_slice:
            for i in range(len(self.snapshot_list)):
                if index_in_file - self.snapshot_list[i].grid_size <= 0:
                    file_index = i

                    # From the end of previous file to beginning of new.
                    if index_in_file == self.snapshot_list[i].grid_size and \
                       is_start:
                        file_index = i+1
                        index_in_file = 0
                    break
                else:
                    index_in_file -= self.snapshot_list[i].grid_size
            return file_index, index_in_file
        else:
            for i in range(len(self.snapshot_list)):
                if index_in_file - self.snapshot_list[i].grid_size < 0:
                    file_index = i
                    break
                else:
                    index_in_file -= self.snapshot_list[i].grid_size
            return file_index, index_in_file

    def __getitem__(self, idx):
        """
        Get an item of the DataSet.

        Parameters
        ----------
        idx : int
            Requested index. NOTE: Slices over multiple files
            are currently NOT supported.

        Returns
        -------
        inputs, outputs : torch.Tensor
            The requested inputs and outputs
        """
        # Get item can be called with an int or a slice.
        if isinstance(idx, int):
            file_index, index_in_file = self._get_file_index(idx)

            # Find out if new data is needed.
            if file_index != self.currently_loaded_file:
                self.get_new_data(file_index)
            return self.input_data[index_in_file], \
                self.output_data[index_in_file]

        elif isinstance(idx, slice):
            # If a slice is requested, we have to find out if it spans files.
            file_index_start, index_in_file_start = self.\
                _get_file_index(idx.start, is_slice=True, is_start=True)
            file_index_stop, index_in_file_stop = self.\
                _get_file_index(idx.stop, is_slice=True)

            # If it does, we cannot deliver.
            # Take care though, if a full snapshot is requested,
            # the stop index will point to the wrong file.
            if file_index_start != file_index_stop:
                if index_in_file_stop == 0:
                    index_in_file_stop = self.snapshot_list[file_index_stop].\
                        grid_size
                else:
                    raise Exception("Lazy loading currently only supports "
                                    "slices in one file. "
                                    "You have requested a slice over two "
                                    "files.")

            # Find out if new data is needed.
            file_index = file_index_start
            if file_index != self.currently_loaded_file:
                self.get_new_data(file_index)
            return self.input_data[index_in_file_start:index_in_file_stop], \
                self.output_data[index_in_file_start:index_in_file_stop]
        else:
            raise Exception("Invalid idx provided.")

    def __len__(self):
        """
        Get the length of the DataSet.

        Returns
        -------
        length : int
            Number of data points in DataSet.
        """
        return self.total_size
