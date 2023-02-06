"""Mixes data between snapshots for improved lazy-loading training."""
import os

import numpy as np

from mala.common.parameters import ParametersData, Parameters
from mala.common.parallelizer import printout
from mala.datahandling.data_handler_base import DataHandlerBase


class DataShuffler(DataHandlerBase):
    """
    Mixes data between snapshots for improved lazy-loading training.

    This is a DISK operation - new, shuffled snapshots will be created on disk.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters used to create the data handling object.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data. If None, then one will
        be created by this class.

    target_calculator : mala.targets.target.Target
        Used to do unit conversion on output data. If None, then one will
        be created by this class.
    """

    def __init__(self, parameters: Parameters, target_calculator=None,
                 descriptor_calculator=None):
        super(DataShuffler, self).__init__(parameters,
                                           target_calculator=target_calculator,
                                           descriptor_calculator=
                                           descriptor_calculator)
        if self.descriptor_calculator.parameters.descriptors_contain_xyz:
            printout("Disabling XYZ-cutting from descriptor data for "
                     "shuffling. If needed, please re-enable afterwards.")
            self.descriptor_calculator.parameters.descriptors_contain_xyz = \
                False

    def add_snapshot(self, input_npy_file, input_npy_directory,
                     output_npy_file, output_npy_directory,
                     snapshot_type="numpy"):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        input_npy_file : string
            File with saved numpy input array.

        input_npy_directory : string
            Directory containing input_npy_directory.

        output_npy_file : string
            File with saved numpy output array.

        output_npy_directory : string
            Directory containing output_npy_file.

        snapshot_type : string
            Either "numpy" or "openpmd" based on what kind of files you
            want to operate on.
        """
        super(DataShuffler, self).\
            add_snapshot(input_npy_file, input_npy_directory,
                         output_npy_file, output_npy_directory,
                         add_snapshot_as="te",
                         output_units="None", input_units="None",
                         calculation_output_file="",
                         snapshot_type=snapshot_type)

    def shuffle_snapshots(self, save_path, save_name="mala_shuffled_snapshot*",
                          shuffle_dimensions=None):
        """
        Shuffle the snapshots into new snapshots.

        This saves them to file.

        Parameters
        ----------
        save_path : string
            Where to save the new, shuffled snapshots.

        save_name : string
            Name of the snapshots to be shuffled.

        shuffle_dimensions : list
            If not None, must be a list of three entries [x,y,z].
            If specified, the shuffling class will use these dimensions
            instead of the ones automatically retrieved from the snapshots.
        """
        # Check the dimensions of the snapshots.
        self._check_snapshots()
        snapshot_size_list = [snapshot.grid_size for snapshot in
                              self.parameters.snapshot_directories_list]

        # Calculate how data will be split.
        if shuffle_dimensions is None:
            # If all snapshots have the same size, we will just re-use
            # this size if not otherwise specified.
            if np.max(snapshot_size_list) == np.min(snapshot_size_list):
                shuffle_dimensions = self.parameters.\
                    snapshot_directories_list[0].grid_dimension
            else:
                # If they do have different sizes, we start with the smallest
                # snapshot, there is some padding down below anyhow.
                shuffle_dimensions = self.parameters.\
                    snapshot_directories_list[np.argmin(snapshot_size_list)].\
                    grid_dimension

        # Padding the new snapshot size until we can fit all snapshots.
        number_of_data_points = np.sum(snapshot_size_list)
        if number_of_data_points % \
                int(np.prod(shuffle_dimensions)) != 0:
            old_shuffle_dimensions = shuffle_dimensions.copy()
            while number_of_data_points % \
                    int(np.prod(shuffle_dimensions)) != 0:
                shuffle_dimensions[3] += 1
            printout("Had to readjust shuffle dimensions from",
                     old_shuffle_dimensions, "to", shuffle_dimensions)

        number_of_new_snapshots = int(number_of_data_points /
                                      int(np.prod(shuffle_dimensions)))
        printout("Data shuffler will generate", number_of_new_snapshots,
                 "new snapshots.")

        # Prepare permutations.
        permutations = []
        for i in range(0, number_of_new_snapshots):

            # This makes the shuffling deterministic, if specified by the user.
            if self.parameters.shuffling_seed is not None:
                np.random.seed(i*self.parameters.shuffling_seed)
            permutations.append(np.random.permutation(
                int(np.prod(shuffle_dimensions))))

        # Load the data (via memmap).
        descriptor_data = []
        target_data = []
        for idx, snapshot in enumerate(self.parameters.
                                               snapshot_directories_list):
            # TODO: Use descriptor and target calculator for this.
            descriptor_data.append(np.load(os.path.join(snapshot.
                                                        input_npy_directory,
                                                        snapshot.input_npy_file),
                                           mmap_mode="r"))
            target_data.append(np.load(os.path.join(snapshot.
                                                    output_npy_directory,
                                                    snapshot.output_npy_file),
                                       mmap_mode="r"))

        # Do the actual shuffling.
        for i in range(0, number_of_new_snapshots):
            new_descriptors = np.zeros((int(np.prod(shuffle_dimensions)),
                                        self.input_dimension))
            new_targets = np.zeros((int(np.prod(shuffle_dimensions)),
                                    self.output_dimension))
            last_start = 0
            snapshot_name = os.path.join(save_path, save_name.replace("*", str(i)))

            # Each new snapshot gets an number_of_new_snapshots-th from each
            # snapshot.
            for j in range(0, self.nr_snapshots):
                current_grid_size = self.parameters.\
                                    snapshot_directories_list[j].grid_size
                current_chunk = int(current_grid_size /
                                    number_of_new_snapshots)
                new_descriptors[last_start:current_chunk+last_start] = \
                    descriptor_data[j].reshape(current_grid_size,
                                               self.input_dimension) \
                    [i*current_chunk:(i+1)*current_chunk, :]
                new_targets[last_start:current_chunk+last_start] = \
                    target_data[j].reshape(current_grid_size,
                                           self.output_dimension) \
                    [i*current_chunk:(i+1)*current_chunk, :]

                last_start += current_chunk

            # Randomize and save to disk.
            new_descriptors = new_descriptors[permutations[i]]
            new_targets = new_targets[permutations[i]]
            new_descriptors = new_descriptors.reshape([shuffle_dimensions[0],
                                                       shuffle_dimensions[1],
                                                       shuffle_dimensions[2],
                                                       self.input_dimension])
            new_targets = new_targets.reshape([shuffle_dimensions[0],
                                               shuffle_dimensions[1],
                                               shuffle_dimensions[2],
                                               self.output_dimension])
            np.save(snapshot_name+".in.npy", new_descriptors)
            np.save(snapshot_name+".out.npy", new_targets)

        # Since no training will be done with this class, we should always
        # clear the data at the end.
        self.clear_data()
