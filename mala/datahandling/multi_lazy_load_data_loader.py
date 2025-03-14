"""Class for loading multiple data sets with pre-fetching."""

import os

import numpy as np
from torch.utils.data import DataLoader

from multiprocessing import shared_memory
import concurrent.futures


class MultiLazyLoadDataLoader:
    """
    Class for loading multiple data sets with pre-fetching.

    Parameters
    ----------
    datasets : list
        A list of data sets to be used in this class.
    """

    def __init__(self, datasets, **kwargs):
        self._datasets = datasets
        self._loaders = []
        for d in datasets:
            self._loaders.append(
                DataLoader(d, batch_size=None, **kwargs, shuffle=False)
            )

        # Create single process pool for prefetching
        # Can use ThreadPoolExecutor for debugging.
        # self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self._pool = concurrent.futures.ProcessPoolExecutor(1)

        # Allocate shared memory and commence file load for first
        # dataset in list
        dset = self._datasets[0]
        dset.allocate_shared_mem()
        self._load_future = self._pool.submit(
            self.load_snapshot_to_shm,
            dset.snapshot,
            dset.descriptor_calculator,
            dset.target_calculator,
            dset.input_shm_name,
            dset.output_shm_name,
        )

    def __len__(self):
        """
        Access number of datasets/snapshots contained within this loader.

        Returns
        -------
        length : int
            Number of datasets/snapshots contained within this loader.
        """
        return len(self._loaders)

    def __iter__(self):
        """
        Generate an iterator.

        Returns
        -------
        iterator: MultiLazyLoadDataLoader
            An iterator over the individual datasets/snapshots in this
            object.
        """
        self._count = 0
        return self

    def __next__(self):
        """
        Access the next data loader.

        Returns
        -------
        iterator: DataLoader
            The next data loader.
        """
        self._count += 1
        if self._count > len(self._loaders):
            raise StopIteration
        else:
            # Wait on last prefetch
            if self._count - 1 >= 0:
                if not self._datasets[self._count - 1].loaded:
                    self._load_future.result()
                    self._datasets[self._count - 1].loaded = True

            # Delete last
            if self._count - 2 >= 0:
                self._datasets[self._count - 2].delete_data()

            # Prefetch next file (looping around epoch boundary)
            dset = self._datasets[self._count % len(self._loaders)]
            if not dset.loaded:
                dset.allocate_shared_mem()
                self._load_future = self._pool.submit(
                    self.load_snapshot_to_shm,
                    dset.snapshot,
                    dset.descriptor_calculator,
                    dset.target_calculator,
                    dset.input_shm_name,
                    dset.output_shm_name,
                )

            # Return current
            return self._loaders[self._count - 1]

    # TODO: Without this function, I get 2 times the number of snapshots
    # memory leaks after shutdown. With it, I get 1 times the number of
    # snapshots. So using this function seems to be correct; just simply not
    # enough? I am not sure where the memory leak is coming from.
    def cleanup(self):
        """Deallocate arrays still left in memory."""
        for dset in self._datasets:
            dset.deallocate_shared_mem()
        self._pool.shutdown()

    # Worker function to load data into shared memory (limited to numpy files
    # only for now)
    @staticmethod
    def load_snapshot_to_shm(
        snapshot,
        descriptor_calculator,
        target_calculator,
        input_shm_name,
        output_shm_name,
    ):
        """
        Load a snapshot into shared memory.

        Limited to numpy files for now.

        Parameters
        ----------
        snapshot : mala.datahandling.snapshot.Snapshot
            Snapshot to be loaded.

        descriptor_calculator : mala.descriptors.descriptor.Descriptor
            Used to do unit conversion on input data.

        target_calculator : mala.targets.target.Target or derivative
            Used to do unit conversion on output data.

        input_shm_name : str
            Name of the input shared memory buffer.

        output_shm_name : str
            Name of the output shared memory buffer.
        """
        # Get shared mem
        input_shm = shared_memory.SharedMemory(name=input_shm_name)
        output_shm = shared_memory.SharedMemory(name=output_shm_name)

        if snapshot.snapshot_type == "numpy":
            input_shape, input_dtype = (
                descriptor_calculator.read_dimensions_from_numpy_file(
                    os.path.join(
                        snapshot.input_npy_directory, snapshot.input_npy_file
                    ),
                    read_dtype=True,
                )
            )

            output_shape, output_dtype = (
                target_calculator.read_dimensions_from_numpy_file(
                    os.path.join(
                        snapshot.output_npy_directory,
                        snapshot.output_npy_file,
                    ),
                    read_dtype=True,
                )
            )
        elif snapshot.snapshot_type == "openpmd":
            input_shape, input_dtype = (
                descriptor_calculator.read_dimensions_from_openpmd_file(
                    os.path.join(
                        snapshot.input_npy_directory, snapshot.input_npy_file
                    ),
                    read_dtype=True,
                )
            )

            output_shape, output_dtype = (
                target_calculator.read_dimensions_from_openpmd_file(
                    os.path.join(
                        snapshot.output_npy_directory,
                        snapshot.output_npy_file,
                    ),
                    read_dtype=True,
                )
            )
        elif snapshot.snapshot_type == "json+openpmd":
            input_shape = descriptor_calculator.read_dimensions_from_json(
                os.path.join(
                    snapshot.input_npy_directory,
                    snapshot.input_npy_file,
                ),
            )
            input_dtype = np.dtype(np.float32)

            output_shape, output_dtype = (
                target_calculator.read_dimensions_from_openpmd_file(
                    os.path.join(
                        snapshot.output_npy_directory,
                        snapshot.output_npy_file,
                    ),
                    read_dtype=True,
                )
            )
        elif snapshot.snapshot_type == "json+numpy":
            input_shape = descriptor_calculator.read_dimensions_from_json(
                os.path.join(
                    snapshot.input_npy_directory,
                    snapshot.input_npy_file,
                ),
            )
            input_dtype = np.dtype(np.float32)

            output_shape, output_dtype = (
                target_calculator.read_dimensions_from_numpy_file(
                    os.path.join(
                        snapshot.output_npy_directory,
                        snapshot.output_npy_file,
                    ),
                    read_dtype=True,
                )
            )
        else:
            raise Exception("Invalid snapshot type selected.")

        # Form numpy arrays from shm buffers
        input_data = np.ndarray(
            shape=input_shape, dtype=input_dtype, buffer=input_shm.buf
        )
        output_data = np.ndarray(
            shape=output_shape, dtype=output_dtype, buffer=output_shm.buf
        )

        # Load numpy data into shm buffers
        if snapshot.snapshot_type == "numpy":
            descriptor_calculator.read_from_numpy_file(
                os.path.join(
                    snapshot.input_npy_directory, snapshot.input_npy_file
                ),
                units=snapshot.input_units,
                array=input_data,
            )
            target_calculator.read_from_numpy_file(
                os.path.join(
                    snapshot.output_npy_directory, snapshot.output_npy_file
                ),
                units=snapshot.output_units,
                array=output_data,
            )

        elif snapshot.snapshot_type == "json+numpy":
            descriptor_calculator.read_from_numpy_file(
                snapshot.temporary_input_file,
                units=snapshot.input_units,
                array=input_data,
            )
            target_calculator.read_from_numpy_file(
                os.path.join(
                    snapshot.output_npy_directory, snapshot.output_npy_file
                ),
                units=snapshot.output_units,
                array=output_data,
            )

        elif snapshot.snapshot_type == "openpmd":
            descriptor_calculator.read_from_openpmd_file(
                os.path.join(
                    snapshot.input_npy_directory, snapshot.input_npy_file
                ),
                units=snapshot.input_units,
                array=input_data,
            )
            target_calculator.read_from_openpmd_file(
                os.path.join(
                    snapshot.output_npy_directory, snapshot.output_npy_file
                ),
                units=snapshot.output_units,
                array=output_data,
            )
        elif snapshot.snapshot_type == "json+openpmd":
            descriptor_calculator.read_from_numpy_file(
                snapshot.temporary_input_file,
                units=snapshot.input_units,
                array=input_data,
            )
            target_calculator.read_from_openpmd_file(
                os.path.join(
                    snapshot.output_npy_directory, snapshot.output_npy_file
                ),
                units=snapshot.output_units,
                array=output_data,
            )
        else:
            raise Exception("Invalid snapshot type selected.")

        # This function only loads the numpy data with scaling. Remaining data
        # preprocessing occurs in __getitem__ of LazyLoadDatasetSingle
        input_shm.close()
        output_shm.close()
