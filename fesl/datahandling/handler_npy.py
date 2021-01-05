from .handler_base import HandlerBase
from .snapshot import Snapshot
import numpy as np
import torch
from torch.utils.data import TensorDataset


class HandlerNpy(HandlerBase):
    """Data handler for *.npy files."""

    def __init__(self, p, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser):
        super(HandlerNpy, self).__init__(p, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser)
        self.input_dimension = 0
        self.output_dimension = 0
        self.dbg_grid_dimensions = p.debug.grid_dimensions

    def add_snapshot(self, input_npy_file, input_npy_directory, output_npy_file, output_npy_directory,
                     input_units="None", output_units="1/eV"):
        """Adds a snapshot to data handler. For this type of data,
        a QuantumEspresso outfile, an outfile from the LDOS calculation and
        a directory containing the cube files"""
        snapshot = Snapshot.as_npy_npy(input_npy_file, input_npy_directory, output_npy_file, output_npy_directory,
                                       input_units, output_units)
        self.parameters.snapshot_directories_list.append(snapshot)

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

            print("Reading descriptors from ", snapshot.input_npy_file, "at", snapshot.input_npy_directory)
            tmp = self.load_from_npy_file(snapshot.input_npy_directory + snapshot.input_npy_file,
                                          mmapmode=self.parameters.input_memmap_mode)

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

            # Now the snapshot input can be converted and added.

            self.raw_input_grid.append(self.descriptor_calculator.convert_units(tmp, snapshot.input_units))

            ####################
            # Targets.
            ####################

            print("Reading targets from ", snapshot.output_npy_file, "at", snapshot.output_npy_directory)
            tmp_out = self.load_from_npy_file(snapshot.output_npy_directory + snapshot.output_npy_file,
                                              mmapmode=self.parameters.output_memmap_mode)

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

            # Now the snapshot output can be added.
            self.raw_output_grid.append(self.target_parser.convert_units(tmp_out, snapshot.output_units))

            if firstsnapshot:
                firstsnapshot = False

        self.grid_size = self.grid_dimension[0] * self.grid_dimension[1] * self.grid_dimension[2]

        self.raw_input_grid = np.array(self.raw_input_grid)
        self.raw_output_grid = np.array(self.raw_output_grid)

        # Before doing anything with the data, we make sure it's float32.
        # Elsewise Pytorch will copy, not reference our data.
        self.raw_input_grid = self.raw_input_grid.astype(np.float32)
        self.raw_output_grid = self.raw_output_grid.astype(np.float32)

        # Reshape from grid to datasize.

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
