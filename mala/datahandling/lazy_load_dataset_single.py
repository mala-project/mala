"""DataSet for lazy-loading."""
import os
from multiprocessing import shared_memory

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LazyLoadDatasetSingle(torch.utils.data.Dataset):
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

    def __init__(self, batch_size, snapshot, input_dimension, output_dimension,
                 input_data_scaler, output_data_scaler, descriptor_calculator,
                 target_calculator, use_horovod,
                 input_requires_grad=False):
        self.snapshot = snapshot
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

        self.batch_size = batch_size
        self.len = int(np.ceil(snapshot.grid_size / self.batch_size))
        self.indices = np.arange(snapshot.grid_size)
        self.input_shm_name = None
        self.output_shm_name = None
        self.loaded = False
        self.allocated = False

    def allocate_shared_mem(self):
        """
        Allocate the shared memory buffer for use by prefetching process.

        Buffer is sized via numpy metadata.
        """
        # Get array shape and data types
        if self.snapshot.snapshot_type == "numpy":
            self.input_shape, self.input_dtype = self.descriptor_calculator. \
                read_dimensions_from_numpy_file(
                os.path.join(self.snapshot.input_npy_directory,
                             self.snapshot.input_npy_file), read_dtype=True)

            self.output_shape, self.output_dtype = self.target_calculator. \
                read_dimensions_from_numpy_file(
                os.path.join(self.snapshot.output_npy_directory,
                             self.snapshot.output_npy_file), read_dtype=True)
        elif self.snapshot.snapshot_type == "openpmd":
            self.input_shape, self.input_dtype = self.descriptor_calculator. \
                read_dimensions_from_openpmd_file(
                os.path.join(self.snapshot.input_npy_directory,
                             self.snapshot.input_npy_file), read_dtype=True)

            self.output_shape, self.output_dtype = self.target_calculator. \
                read_dimensions_from_openpmd_file(
                os.path.join(self.snapshot.output_npy_directory,
                             self.snapshot.output_npy_file), read_dtype=True)
        else:
            raise Exception("Invalid snapshot type selected.")

        # To avoid copies and dealing with in-place casting from FP64, restrict
        # usage to data in FP32 type (which is a good idea anyway to save
        # memory)
        if self.input_dtype != np.float32 or self.output_dtype != np.float32:
            raise Exception("LazyLoadDatasetSingle requires numpy data in "
                            "FP32.")

        # Allocate shared memory buffer
        input_bytes = self.input_dtype.itemsize * np.prod(self.input_shape)
        output_bytes = self.output_dtype.itemsize * np.prod(self.output_shape)
        input_shm = shared_memory.SharedMemory(create=True, size=input_bytes)
        output_shm = shared_memory.SharedMemory(create=True, size=output_bytes)

        self.input_shm_name = input_shm.name
        self.output_shm_name = output_shm.name

        input_shm.close()
        output_shm.close()
        self.allocated = True

    def deallocate_shared_mem(self):
        """Deallocate the shared memory buffer used by prefetching process."""
        if self.allocated:
            input_shm = shared_memory.SharedMemory(name=self.input_shm_name)
            output_shm = shared_memory.SharedMemory(name=self.output_shm_name)
            input_shm.close()
            input_shm.unlink()

            output_shm.close()
            output_shm.unlink()

        self.allocated = False

    def delete_data(self):
        """Free the shared memory buffers."""
        if self.loaded:
            self.input_data = None
            self.output_data = None
            self.deallocate_shared_mem()
        self.loaded = False

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
        input_shm = shared_memory.SharedMemory(name=self.input_shm_name)
        output_shm = shared_memory.SharedMemory(name=self.output_shm_name)

        input_data = np.ndarray(shape=[self.snapshot.grid_size,
                                       self.input_dimension],
                                dtype=np.float32, buffer=input_shm.buf)
        output_data = np.ndarray(shape=[self.snapshot.grid_size,
                                        self.output_dimension],
                                 dtype=np.float32, buffer=output_shm.buf)
        if idx == self.len-1:
            batch = self.indices[idx * self.batch_size:]
        else:
            batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        # print(batch.shape)

        input_batch = input_data[batch, ...]
        output_batch = output_data[batch, ...]

        # Perform conversion to tensor and perform transforms
        input_batch = torch.from_numpy(input_batch)
        self.input_data_scaler.transform(input_batch)
        input_batch.requires_grad = self.input_requires_grad

        if self.return_outputs_directly is False:
            output_batch = torch.from_numpy(output_batch)
            self.output_data_scaler.transform(output_batch)

        input_shm.close()
        output_shm.close()

        return input_batch, output_batch

    def __len__(self):
        """
        Get the length of the DataSet.

        Returns
        -------
        length : int
            Number of data points in DataSet.
        """
        return self.len

    def mix_datasets(self):
        """
        Shuffle the data in this data set.

        For this class, instead of mixing the datasets, we just shuffle the
        indices and leave the dataset order unchanged.
        NOTE: It seems that the shuffled access to the shared memory
        performance is much reduced (relative to the FastTensorDataset).
        To regain performance, can rewrite to shuffle the datasets like in
        the existing LazyLoadDataset. Another option might be
        to try loading the numpy file in permuted order to avoid the shuffled
        reads; however, this might require some care to
        avoid erroneously overwriting shared memory data in cases where a
        single dataset object is used back to back.
        """
        np.random.shuffle(self.indices)

