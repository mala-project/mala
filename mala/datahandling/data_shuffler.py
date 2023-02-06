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

    def add_snapshot(self, input_file, input_directory,
                     output_file, output_directory,
                     snapshot_type="numpy"):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        input_file : string
            File with saved numpy input array.

        input_directory : string
            Directory containing input_npy_directory.

        output_file : string
            File with saved numpy output array.

        output_directory : string
            Directory containing output_npy_file.

        snapshot_type : string
            Either "numpy" or "openpmd" based on what kind of files you
            want to operate on.
        """
        super(DataShuffler, self).\
            add_snapshot(input_file, input_directory,
                         output_file, output_directory,
                         add_snapshot_as="te",
                         output_units="None", input_units="None",
                         calculation_output_file="",
                         snapshot_type=snapshot_type)

    def shuffle_snapshots(self,
                          complete_save_path=None,
                          descriptor_save_path=None,
                          target_save_path=None,
                          save_name="mala_shuffled_snapshot*",
                          number_of_shuffled_snapshots=None):
        """
        Shuffle the snapshots into new snapshots.

        This saves them to file.

        Parameters
        ----------
        complete_save_path : string
            If not None: the directory in which all snapshots will be saved.
            Overwrites descriptor_save_path, target_save_path and
            additional_info_save_path if set.

        descriptor_save_path : string
            Directory in which to save descriptor data.

        target_save_path : string
            Directory in which to save target data.

        save_name : string
            Name of the snapshots to be shuffled.

        number_of_shuffled_snapshots : int
            If not None, this class will attempt to redistribute the data
            to this amount of snapshots. If None, then the same number of
            snapshots provided will be used.
        """
        # Check the paths.
        if complete_save_path is not None:
            descriptor_save_path = complete_save_path
            target_save_path = complete_save_path
        else:
            if target_save_path is None or descriptor_save_path is None:
                raise Exception("No paths to save shuffled data provided.")

        # Check the file format.
        if "." in save_name:
            file_ending = save_name.split(".")[-1]
            save_name = save_name.split(".")[0]
            if file_ending != "npy":
                import openpmd_api as io

                if file_ending not in io.file_extensions:
                    raise Exception("Invalid file ending selected: " +
                                    file_ending)
        else:
            file_ending = "npy"

        # Check the dimensions of the snapshots.
        self._check_snapshots()
        snapshot_size_list = [snapshot.grid_size for snapshot in
                              self.parameters.snapshot_directories_list]
        number_of_data_points = np.sum(snapshot_size_list)

        if number_of_shuffled_snapshots is None:
            # If the user does not tell us how many snapshots to use,
            # we have to check if the number of snapshots is straightforward.
            # If all snapshots have the same size, we can just replicate the
            # snapshot structure.
            if np.max(snapshot_size_list) == np.min(snapshot_size_list):
                shuffle_dimensions = self.parameters.\
                    snapshot_directories_list[0].grid_dimension
                number_of_new_snapshots = self.nr_snapshots
            else:
                # If the snapshots have different sizes we simply create
                # (x, 1, 1) snapshots big enough to hold the data.
                number_of_new_snapshots = self.nr_snapshots
                while number_of_data_points % number_of_new_snapshots != 0:
                    number_of_new_snapshots += 1

                # If they do have different sizes, we start with the smallest
                # snapshot, there is some padding down below anyhow.
                shuffle_dimensions = [int(number_of_data_points /
                                          number_of_new_snapshots), 1, 1]
        else:
            number_of_new_snapshots = number_of_shuffled_snapshots
            if number_of_data_points % number_of_new_snapshots != 0:
                raise Exception("Cannot create this number of snapshots "
                                "from data provided.")
            else:
                shuffle_dimensions = [int(number_of_data_points /
                                          number_of_new_snapshots), 1, 1]

        printout("Data shuffler will generate", number_of_new_snapshots,
                 "new snapshots.")
        printout("Shuffled snapshot dimension will be ", shuffle_dimensions)

        # Prepare permutations.
        permutations = []
        seeds = []
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
            descriptor_name = os.path.join(descriptor_save_path,
                                           save_name.replace("*", str(i)))
            target_name = os.path.join(target_save_path,
                                       save_name.replace("*", str(i)))

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
            if file_ending == "npy":
                self.descriptor_calculator.\
                    write_to_numpy_file(descriptor_name+".in.npy",
                                        new_descriptors)
                self.target_calculator.\
                    write_to_numpy_file(target_name+".out.npy",
                                        new_targets)
            else:
                # We check above that in the non-numpy case, OpenPMD will work.
                self.descriptor_calculator.grid_dimensions = shuffle_dimensions
                self.target_calculator.grid_dimensions = shuffle_dimensions
                self.descriptor_calculator.\
                    write_to_openpmd_file(descriptor_name+".in."+file_ending,
                                          new_descriptors,
                                          additional_attributes={"global_shuffling_seed": self.parameters.shuffling_seed,
                                                                 "local_shuffling_seed": i*self.parameters.shuffling_seed})
                self.target_calculator.\
                    write_to_openpmd_file(target_name+".out."+file_ending,
                                          target_data=new_targets,
                                          additional_attributes={"global_shuffling_seed": self.parameters.shuffling_seed,
                                                                 "local_shuffling_seed": i*self.parameters.shuffling_seed})

        # Since no training will be done with this class, we should always
        # clear the data at the end.
        self.clear_data()
