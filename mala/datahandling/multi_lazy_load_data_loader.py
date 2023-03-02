import os

import numpy as np
from torch.utils.data import DataLoader

from multiprocessing import shared_memory
import concurrent.futures

# Class to handle loading multiple LazyLoadDatasetSingle objects with
# prefetching


class MultiLazyLoadDataLoader:
    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.loaders = []
        for d in datasets:
            self.loaders.append(DataLoader(d,
                                          batch_size=None,
                                          **kwargs,
                                          shuffle=False))

        # Create single process pool for prefetching
        # Can use ThreadPoolExecutor for debugging.
        #self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.pool = concurrent.futures.ProcessPoolExecutor(1)

        # Allocate shared memory and commence file load for first
        # dataset in list
        dset = self.datasets[0]
        dset.allocate_shared_mem()
        self.load_future = self.pool.submit(self.load_snapshot_to_shm,
                                            dset.snapshot,
                                            dset.descriptor_calculator,
                                            dset.target_calculator,
                                            dset.input_shm_name,
                                            dset.output_shm_name)

    def __len__(self):
        return len(self.loaders)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count > len(self.loaders):
            raise StopIteration
        else:
            # Wait on last prefetch
            if self.count - 1 >= 0:
                if not self.datasets[self.count - 1].loaded:
                    self.load_future.result()
                    self.datasets[self.count - 1].loaded = True

            # Delete last
            if self.count - 2 >= 0:
                self.datasets[self.count - 2].delete_data()

            # Prefetch next file (looping around epoch boundary)
            dset = self.datasets[self.count % len(self.loaders)]
            if not dset.loaded:
              dset.allocate_shared_mem()
              self.load_future = self.pool.submit(self.load_snapshot_to_shm,
                                                  dset.snapshot,
                                                  dset.descriptor_calculator,
                                                  dset.target_calculator,
                                                  dset.input_shm_name,
                                                  dset.output_shm_name)

            # Return current
            return self.loaders[self.count - 1]

    # Worker function to load data into shared memory (limited to numpy files
    # only for now)
    @staticmethod
    def load_snapshot_to_shm(snapshot, descriptor_calculator, target_calculator,
                             input_shm_name, output_shm_name):
        # Get shared mem
        input_shm = shared_memory.SharedMemory(name=input_shm_name)
        output_shm = shared_memory.SharedMemory(name=output_shm_name)

        input_shape, input_dtype = descriptor_calculator. \
            read_dimensions_and_dtype_from_numpy_file(
            os.path.join(snapshot.input_npy_directory,
                         snapshot.input_npy_file))

        output_shape, output_dtype = target_calculator. \
            read_dimensions_and_dtype_from_numpy_file(
            os.path.join(snapshot.output_npy_directory,
                         snapshot.output_npy_file))

        # Form numpy arrays from shm buffers
        input_data = np.ndarray(shape=input_shape, dtype=input_dtype,
                                buffer=input_shm.buf)
        output_data = np.ndarray(shape=output_shape, dtype=output_dtype,
                                 buffer=output_shm.buf)

        # Load numpy data into shm buffers
        descriptor_calculator. \
            read_from_numpy_file(
            os.path.join(snapshot.input_npy_directory,
                         snapshot.input_npy_file),
            units=snapshot.input_units,
            array=input_data)
        target_calculator. \
            read_from_numpy_file(
            os.path.join(snapshot.output_npy_directory,
                         snapshot.output_npy_file),
            units=snapshot.output_units,
            array=output_data)

        # This function only loads the numpy data with scaling. Remaining data
        # preprocessing occurs in __getitem__ of LazyLoadDatasetSingle
        input_shm.close()
        output_shm.close()
