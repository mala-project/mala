from .handler_base import HandlerBase
import numpy as np
import torch
from torch.utils.data import TensorDataset


class HandlerNpy(HandlerBase):
    """Data handler for *.npy files."""

    def __init__(self, p, input_data_scaler, output_data_scaler):
        super(HandlerNpy, self).__init__(p, input_data_scaler, output_data_scaler)
        self.input_dimension = 0
        self.output_dimension = 0
        self.dbg_grid_dimensions = p.debug.grid_dimensions

    def add_snapshot(self, npy_input_file, npy_input_directory, npy_output_file, npy_output_directory):
        """Adds a snapshot to data handler. For this type of data,
        a QuantumEspresso outfile, an outfile from the LDOS calculation and
        a directory containing the cube files"""
        self.parameters.snapshot_directories_list.append(
            [npy_input_file, npy_input_directory, npy_output_file, npy_output_directory])

    def load_data(self):
        """Loads the data from saved numpy arrays into RAM"""

        # determine number of snapshots.
        self.nr_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots into RAM.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            print("Reading descriptors from ", snapshot[0], "at", snapshot[1])
            tmp = self.load_from_npy_file(snapshot[1] + snapshot[0], mmapmode=self.parameters.input_memmap_mode)

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
                    raise Exception("Invalid snapshot entered at ", snapshot[0])

            # Now the snapshot input can be added.
            self.raw_input_grid.append(tmp)

            ####################
            # Targets.
            ####################

            print("Reading targets from ", snapshot[2], "at", snapshot[3])
            tmp_out = self.load_from_npy_file(snapshot[3] + snapshot[2], mmapmode=self.parameters.output_memmap_mode)

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = np.shape(tmp_out)[-1]
            tmp_grid_dim = np.shape(tmp_out)[0:3]
            if firstsnapshot:
                self.output_dimension = tmp_output_dimension
            else:
                if self.output_dimension != tmp_output_dimension:
                    raise Exception("Invalid snapshot entered at ", snapshot[2])
            if (self.grid_dimension[0] != tmp_grid_dim[0]
                    or self.grid_dimension[1] != tmp_grid_dim[1]
                    or self.grid_dimension[2] != tmp_grid_dim[2]):
                raise Exception("Invalid snapshot entered at ", snapshot[2])

            # Now the snapshot output can be added.
            self.raw_output_grid.append(tmp_out)

            if firstsnapshot:
                firstsnapshot = False

        self.grid_size = self.grid_dimension[0] * self.grid_dimension[1] * self.grid_dimension[2]

        self.raw_input_grid = np.array(self.raw_input_grid)
        self.raw_output_grid = np.array(self.raw_output_grid)

        # Before doing anything with the data, we make sure it's float32.
        # Elsewise Pytorch will copy, not reference our data.
        self.raw_input_grid = self.raw_input_grid.astype(np.float32)
        self.raw_output_grid = self.raw_output_grid.astype(np.float32)

        # Reshape from grid to datasize..

        self.grid_to_datasize()


    def load_from_npy_file(self, file, mmapmode=None):
        loaded_array = np.load(file, mmap_mode=mmapmode)
        if len(self.dbg_grid_dimensions) == 3:
            try:
                return loaded_array[0:self.dbg_grid_dimensions[0], 0:self.dbg_grid_dimensions[1], 0:self.dbg_grid_dimensions[2], :]
            except:
                print(
                    "Could not use grid reduction, falling back to regular grid. Please check that the debug grid is "
                    "not bigger than the actual grid.")

        else:
            return loaded_array

    def get_input_dimension(self):
        if self.input_dimension > 0:
            return self.input_dimension
        else:
            raise Exception("No descriptors were calculated, cannot give input dimension.")

    def get_output_dimension(self):
        if self.output_dimension > 0:
            return self.output_dimension
        else:
            raise Exception("No targets were read, cannot give output dimension.")
