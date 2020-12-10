'''
Collects data from saved numpy arrays.
'''


from .handler_base import handler_base
import numpy as np
import torch
from torch.utils.data import TensorDataset


class handler_npy(handler_base):
    """Data handler for *.npy files."""
    def __init__(self, p):
        super(handler_npy,self).__init__(p)
        self.input_dimension = 0
        self.output_dimension = 0
        self.grid_dimension = np.array([0,0,0])
        self.dbg_grid_dimensions = p.debug.grid_dimensions
        datacount = 0

    def add_snapshot(self, npy_input_file, npy_input_directory, npy_output_file, npy_output_directory):
        """Adds a snapshot to data handler. For this type of data,
        a QuantumEspresso outfile, an outfile from the LDOS calculation and
        a directory containing the cube files"""
        self.parameters.snapshot_directories_list.append([npy_input_file, npy_input_directory, npy_output_file, npy_output_directory])

    def load_data(self):
        """Loads the data from saved numpy arrays into RAM"""

        # determine number of snapshots.
        nr_of_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots into RAM.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            print("Reading descriptors from ", snapshot[0], "at", snapshot[1])
            tmp = self.load_from_npy_file(snapshot[1]+snapshot[0])
            # Remove first 3 elements of descriptors, as they correspond
            # to the x,y and z information.
            tmp = tmp[:, :, :, 3:]

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_input_dimension = np.shape(tmp)[-1]
            tmp_grid_dim = np.shape(tmp)[0:3]
            if firstsnapshot == True:
                self.input_dimension = tmp_input_dimension
                self.grid_dimension[0:3] = tmp_grid_dim[0:3]
            else:
                if (self.input_dimension != tmp_input_dimension
                    or self.grid_dimension[0] != tmp_grid_dim[0]
                    or self.grid_dimension[1] != tmp_grid_dim[1]
                    or self.grid_dimension[2] != tmp_grid_dim[2]):
                    raise Exception("Invalid snapshot entered at ", snapshot[0])


            # Now the snapshot input can be added.
            self.raw_input.append(tmp)

            ####################
            # Targets.
            ####################

            print("Reading targets from ", snapshot[2], "at", snapshot[3])
            tmp_out = self.load_from_npy_file(snapshot[3]+snapshot[2])

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = np.shape(tmp_out)[-1]
            tmp_grid_dim = np.shape(tmp_out)[0:3]
            if firstsnapshot == True:
                self.output_dimension = tmp_output_dimension
            else:
                if (self.output_dimension != tmp_output_dimension):
                    raise Exception("Invalid snapshot entered at ", snapshot[2])
            if (self.grid_dimension[0] != tmp_grid_dim[0]
                or self.grid_dimension[1] != tmp_grid_dim[1]
                or self.grid_dimension[2] != tmp_grid_dim[2]):
                raise Exception("Invalid snapshot entered at ", snapshot[2])

            # Now the snapshot output can be added.
            self.raw_output.append(tmp_out)

            if firstsnapshot == True:
                firstsnapshot = False

        self.raw_input = np.array(self.raw_input)
        self.raw_output = np.array(self.raw_output)

        # All snapshots are in RAM now. Now we need to reshape into one long vector.
        # We want to go from
        # number_of_snapshots x gridx x gridy x gridz x feature_length
        # to
        # (number_of_snapshots x gridx x gridy x gridz) x feature_length
        datacount = self.grid_dimension[0]*self.grid_dimension[1]*self.grid_dimension[2]*nr_of_snapshots
        self.raw_input = self.raw_input.reshape([datacount, self.get_input_dimension()])
        self.raw_output = self.raw_output.reshape([datacount, self.get_output_dimension()])

    def prepare_data(self):
        # TODO: Clustering of the data before splitting it up.
        np.random.shuffle(self.raw_input)
        np.random.shuffle(self.raw_output)
        self.raw_input = torch.from_numpy(self.raw_input).float()
        self.raw_output = torch.from_numpy(self.raw_output).float()

        if sum(self.parameters.data_splitting) > 100:
            raise Exception("Will not attempt to use more than 100% of data.")

        # Split data according to parameters.
        # raw_input and raw_ouput are guaranteed to have the same first dimensions
        # as they are calculated using the grid and the number of snapshots.
        # We enforce that the grid size is equal in load_data().
        self.nr_training_data = int(self.parameters.data_splitting[0]/100 *np.shape(self.raw_input)[0])
        self.nr_validation_data = int(self.parameters.data_splitting[1]/100 *np.shape(self.raw_input)[0])
        self.nr_test_data = int(self.parameters.data_splitting[2]/100 *np.shape(self.raw_input)[0])

        # We need to make sure that really all of the data is used.
        missing_data = (self.nr_training_data+self.nr_validation_data+self.nr_test_data)-np.shape(self.raw_input)[0]
        self.nr_test_data += missing_data

        index1 = self.nr_training_data
        index2 = self.nr_training_data+self.nr_validation_data

        self.training_data_set = TensorDataset(self.raw_input[0:2], self.raw_output[0:2])
        self.validation_data_set = TensorDataset(self.raw_input[index1:index2], self.raw_output[index1:index2])
        self.test_data_set = TensorDataset(self.raw_input[index2:], self.raw_output[index2:])


    def load_from_npy_file(self, file):
        loaded_array = np.load(file)
        if (len(self.dbg_grid_dimensions) == 3):
            np.save(file+"small",loaded_array[0:self.dbg_grid_dimensions[0],0:self.dbg_grid_dimensions[1],0:self.dbg_grid_dimensions[2],:])
            try:
                return loaded_array[0:self.dbg_grid_dimensions[0],0:self.dbg_grid_dimensions[1],0:self.dbg_grid_dimensions[2],:]
            except:
                print("Could not use grid reduction, falling back to regular grid. Please check that the debug grid is not bigger than the actual grid.")

        else:
            return loaded_array

    def get_input_dimension(self):
        if (self.input_dimension > 0):
            return self.input_dimension
        else:
            raise Exception("No descriptors were calculated, cannot give input dimension.")

    def get_output_dimension(self):
        if (self.output_dimension > 0):
            return self.output_dimension
        else:
            raise Exception("No targets were read, cannot give output dimension.")
