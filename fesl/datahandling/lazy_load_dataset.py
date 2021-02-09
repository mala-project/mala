import torch
from torch.utils.data import Dataset
from fesl.datahandling.snapshot import Snapshot
from fesl.common.printout import printout
import numpy as np
import random
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass

class LazyLoadDataset(torch.utils.data.Dataset):
    def __init__(self, input_dimension, output_dimension, input_data_scaler, output_data_scaler, descriptor_calculator,
                 target_calculator, grid_dimensions, grid_size, descriptors_contain_xyz, use_horovod):
        self.snapshot_list = []
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler
        self.descriptor_calculator = descriptor_calculator
        self.target_calculator = target_calculator
        self.grid_dimensions = grid_dimensions
        self.grid_size = grid_size
        self.number_of_snapshots = 0
        self.total_size = 0
        self.descriptors_contain_xyz = descriptors_contain_xyz
        self.currently_loaded_file = 0
        self.input_data = np.empty(0)
        self.output_data = np.empty(0)
        self.use_horovod = use_horovod

    def add_snapshot_to_dataset(self, snapshot: Snapshot):
        self.snapshot_list.append(snapshot)
        self.number_of_snapshots += 1
        self.total_size = self.number_of_snapshots*self.grid_size

    def prepare_datasets(self):
        used_perm = torch.randperm(self.number_of_snapshots)
        if self.use_horovod:
            hvd.allreduce(torch.tensor(0), name='barrier')
            used_perm = hvd.broadcast(used_perm, 0)
        self.snapshot_list = [self.snapshot_list[i] for i in used_perm]
        self.get_new_data(0)


    def get_new_data(self, file_index):
        # Load the data into RAM.
        self.input_data = np.load(self.snapshot_list[file_index].input_npy_directory+self.snapshot_list[file_index].input_npy_file)
        self.output_data = np.load(self.snapshot_list[file_index].output_npy_directory+self.snapshot_list[file_index].output_npy_file)

        # Transform the data.
        if self.descriptors_contain_xyz:
            self.input_data = self.input_data[:, :, :, 3:]
        self.input_data = self.input_data.reshape([self.grid_size, self.input_dimension])
        self.input_data *= self.descriptor_calculator.convert_units(1, self.snapshot_list[file_index].input_units)
        self.input_data = self.input_data.astype(np.float32)
        self.input_data = torch.from_numpy(self.input_data).float()
        self.input_data = self.input_data_scaler.transform(self.input_data)


        self.output_data = self.output_data.reshape([self.grid_size, self.output_dimension])
        self.output_data *= self.target_calculator.convert_units(1, self.snapshot_list[file_index].output_units)
        self.output_data = np.array(self.output_data)
        self.output_data = self.output_data.astype(np.float32)
        self.output_data = torch.from_numpy(self.output_data).float()
        self.output_data = self.output_data_scaler.transform(self.output_data)

        # Save which data we have currently loaded.
        self.currently_loaded_file = file_index
        # if self.number_of_snapshots > 1:
        #     print(hvd.rank(), file_index)




    def __getitem__(self, idx):
        # Get item can be called with an int or a slice.
        # slice_is_requested = False
        # if isinstance(idx, int):
        file_index = idx // self.grid_size
        index_in_file = idx % self.grid_size
        if file_index != self.currently_loaded_file:
            self.get_new_data(file_index)
        #
        # elif isinstance(idx, slice):
        #     print("lazy loading requested a slice!")
        #     slice_is_requested = True
        #     file_index_start = idx.start // self.grid_size
        #     index_in_file_start = idx.start % self.grid_size
        #     file_index_stop = idx.stop // self.grid_size
        #     index_in_file_stop = idx.stop % self.grid_size
        #     if self.snapshot_list[file_index_start].input_units != self.snapshot_list[file_index_stop].input_units or \
        #         self.snapshot_list[file_index_start].output_units != self.snapshot_list[file_index_stop].output_units:
        #         raise Exception("Cannot work with snapshots that have different units.")
        #     else:
        #         file_index = file_index_start
        # else:
        #     raise Exception("Unsupported argument, idx needs to be of type int or slice.")

        # Open the files that are needed and get the data out of it. Keep the RAM footprint minimal.
        # if slice_is_requested:
        #     tmp_input = np.zeros((idx.stop-idx.start, self.input_dimension), dtype=np.float32)
        #     tmp_output = np.zeros((idx.stop-idx.start, self.output_dimension), dtype=np.float32)
        #
        #     # Depending on what sample was requested, we might have to open two files.
        #     # In any case we have to open the files, reshape the resulting arrays so that we can work with the index
        #     # and then fill the arrays that are then processed into tensors.
        #     from_file_1 = np.load(self.snapshot_list[file_index_start].input_npy_directory+self.snapshot_list[file_index_start].input_npy_file, mmap_mode='r')
        #     from_file_1 = np.array(from_file_1)
        #     if self.descriptors_contain_xyz:
        #         from_file_1 = from_file_1[:, :, :, 3:]
        #     from_file_1 = from_file_1.reshape([self.grid_size, self.input_dimension])
        #     if file_index_start != file_index_stop:
        #         from_file_2 = np.load(self.snapshot_list[file_index_stop].input_npy_directory+self.snapshot_list[file_index_stop].input_npy_file, mmap_mode='r')
        #         from_file_2 = np.array(from_file_2)
        #         if self.descriptors_contain_xyz:
        #             from_file_2 = from_file_2[:, :, :, 3:]
        #         from_file_2 = from_file_2.reshape([self.grid_size, self.input_dimension])
        #         tmp_input[0:self.grid_size-index_in_file_start] = from_file_1[index_in_file_start:self.grid_size]
        #         tmp_input[self.grid_size-index_in_file_start:idx.stop] = from_file_2[0:index_in_file_stop]
        #     else:
        #         tmp_input[0:idx.stop] = from_file_1[index_in_file_start:index_in_file_stop]
        #
        #     # Depending on what sample was requested, we might have to open two files.
        #     # In any case we have to open the files, reshape the resulting arrays so that we can work with the index
        #     # and then fill the arrays that are then processed into tensors.
        #     from_file_1 = np.load(self.snapshot_list[file_index_start].output_npy_directory+self.snapshot_list[file_index_start].output_npy_file, mmap_mode='r')
        #     from_file_1 = np.array(from_file_1)
        #     from_file_1 = from_file_1.reshape([self.grid_size, self.output_dimension])
        #     if file_index_start != file_index_stop:
        #         from_file_2 = np.load(self.snapshot_list[file_index_stop].output_npy_directory+self.snapshot_list[file_index_stop].output_npy_file, mmap_mode='r')
        #         from_file_2 = np.array(from_file_2)
        #         from_file_2 = from_file_2.reshape([self.grid_size, self.output_dimension])
        #         tmp_output[0:self.grid_size-index_in_file_start] = from_file_1[index_in_file_start:self.grid_size]
        #         tmp_output[self.grid_size-index_in_file_start:idx.stop] = from_file_2[0:index_in_file_stop]
        #     else:
        #         tmp_output[0:idx.stop] = from_file_1[index_in_file_start:index_in_file_stop]
        #
        # else:
        # from_file_1 = np.load(self.snapshot_list[file_index].input_npy_directory+self.snapshot_list[file_index].input_npy_file, mmap_mode='r')
        # from_file_1 = np.array(from_file_1)
        # from_file_1 = from_file_1.reshape([self.grid_size, self.input_dimension])
        #
        # from_file_1 = np.load(self.snapshot_list[file_index].output_npy_directory+self.snapshot_list[file_index].output_npy_file, mmap_mode='r')
        # from_file_1 = np.array(from_file_1)
        # from_file_1 = from_file_1.reshape([self.grid_size, self.output_dimension])


        # All done, we got the data, now preprocess the data for usage in a network.
        # TODO: Put all this into some kind of function.
        # tmp_input = self.input_data[index_in_file]
        # tmp_input *= self.descriptor_calculator.convert_units(1, self.snapshot_list[file_index].input_units)
        # tmp_input = np.array(tmp_input)
        # tmp_input = tmp_input.astype(np.float32)
        # tmp_input = torch.from_numpy(tmp_input).float()
        # tmp_input = self.input_data_scaler.transform(tmp_input)
        #
        # tmp_output = self.output_data[index_in_file]
        # tmp_output *= self.target_calculator.convert_units(1, self.snapshot_list[file_index].output_units)
        # tmp_output = np.array(tmp_output)
        # tmp_output = tmp_output.astype(np.float32)
        # tmp_output = torch.from_numpy(tmp_output).float()
        # tmp_output = self.output_data_scaler.transform(tmp_output)


        return self.input_data[index_in_file], self.output_data[index_in_file]

    def __len__(self):
        return self.total_size
