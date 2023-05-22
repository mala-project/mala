"""Mixes data between snapshots for improved lazy-loading training."""
import os

import numpy as np

from mala.common.parameters import ParametersData, Parameters
from mala.common.parallelizer import printout
from mala.common.physical_data import PhysicalData
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

    def __shuffle_numpy(self, number_of_new_snapshots, shuffle_dimensions,
                        descriptor_save_path, save_name, target_save_path,
                        permutations, file_ending):
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

    class DescriptorOrTarget:

        def __init__(self, save_path, npy_directory, npy_file, calculator,
                     name_infix, dimension):
            self.save_path = save_path
            self.npy_directory = npy_directory
            self.npy_file = npy_file
            self.calculator = calculator
            self.name_infix = name_infix
            self.dimension = dimension


    def __shuffle_openpmd(self, dot, number_of_new_snapshots,
                          shuffle_dimensions, save_name, permutations,
                          file_ending):
        import openpmd_api as io
        # Load the data
        input_series_list = []
        for idx, snapshot in enumerate(
                self.parameters.snapshot_directories_list):
            # TODO: Use descriptor and target calculator for this.
            input_series_list.append(
                io.Series(
                    os.path.join(dot.npy_directory(snapshot),
                                 dot.npy_file(snapshot)), io.Access.read_only))

        # Peek into the input snapshots to determine the datatypes.
        for series in input_series_list:
            for _, iteration in series.iterations.items():
                mesh_out = iteration.meshes[dot.calculator.data_name]
                feature_size = len(mesh_out)
                for _, component in mesh_out.items():
                    dataset = io.Dataset(component.dtype, shuffle_dimensions)
                    break
                break
            break

        # Input datasets are split into n slices where n is the number of output
        # (shuffled) checkpoints.
        # This gets the offset and extent of the i'th such slice.
        # The extent is given as in openPMD, i.e. the size of the block
        # (not its upper coordinate).
        def from_chunk_i(i, n, dset, slice_dimension=0):
            if isinstance(dset, io.Dataset):
                dset = dset.extent
            offset = [0 for _ in dset]
            extent = dset
            extent_dim_0 = dset[slice_dimension]
            if extent_dim_0 % n != 0:
                raise Exception(
                    "Dataset {} cannot be split into {} chunks on dimension {}."
                    .format(dset, n, slice_dimension))
            single_chunk_len = extent_dim_0 // n
            offset[slice_dimension] = i * single_chunk_len
            extent[slice_dimension] = single_chunk_len
            return offset, extent

        # This determines the placement of the j'th of n chunks into a flattened
        # 1D array of a certain length by offset and extent.
        # The extent is given as in Python slices, i.e. the upper coordinate of
        # the block (not its size).
        def to_chunk_j(j, n, length):
            if length % n != 0:
                raise Exception("Cannot split {} items into {} chunks.".format(
                    n, j))
            single_chunk_len = length // n
            return j * single_chunk_len, (j + 1) * single_chunk_len

        # Do the actual shuffling.
        # TODO: Parallelize into `number_of_new_snapshots` workers
        # Each worker would write the i'th snapshot in serial, the input
        # snapshots would be opened one after another in parallel
        for i in range(0, number_of_new_snapshots):
            # We check above that in the non-numpy case, OpenPMD will work.
            dot.calculator.grid_dimensions = shuffle_dimensions
            name_prefix = os.path.join(dot.save_path,
                                       save_name.replace("*", str(i)))
            shuffled_snapshot_series = dot.calculator.\
                    write_to_openpmd_file(name_prefix+dot.name_infix+file_ending,
                                          PhysicalData.SkipArrayWriting(dataset, feature_size),
                                          additional_attributes={"global_shuffling_seed": self.parameters.shuffling_seed,
                                                                 "local_shuffling_seed": i*self.parameters.shuffling_seed})
            mesh_out = shuffled_snapshot_series.write_iterations()[0].meshes[
                dot.calculator.data_name]
            new_array = np.zeros(
                (dot.dimension, int(np.prod(shuffle_dimensions))))

            for j in range(0, self.nr_snapshots):
                mesh_in = input_series_list[j].iterations[0].meshes[
                    dot.calculator.data_name]

                from_chunk_offset, from_chunk_extent = from_chunk_i(
                    i, number_of_new_snapshots, dataset)
                to_chunk_offset, to_chunk_extent = to_chunk_j(
                    j, number_of_new_snapshots, np.prod(shuffle_dimensions))
                for dimension in range(len(mesh_in)):
                    mesh_in[str(dimension)].load_chunk(
                        new_array[dimension, to_chunk_offset:to_chunk_extent],
                        from_chunk_offset, from_chunk_extent)
                mesh_in.series_flush()

            for k in range(feature_size):
                rc = mesh_out[str(k)]
                rc[:, :, :] = new_array[k, :][permutations[i]].reshape(
                    shuffle_dimensions)
            shuffled_snapshot_series.close()


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

        # TODO: This opens each input Series, change this for parallelization
        # Check the dimensions of the snapshots.
        self._check_snapshots()

        snapshot_types = {
            snapshot.snapshot_type
            for snapshot in self.parameters.snapshot_directories_list
        }
        if len(snapshot_types) > 1:
            raise Exception(
                "[data_shuffler] Can only deal with one type of input snapshot"
                + " at once (openPMD or numpy).")
        snapshot_type = snapshot_types.pop()
        del snapshot_types

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
            if snapshot_type == 'openpmd':
                import math
                import functools
                number_of_new_snapshots = functools.reduce(
                    math.gcd, [
                        snapshot.grid_dimension[0] for snapshot in
                        self.parameters.snapshot_directories_list
                    ], number_of_new_snapshots)
        else:
            number_of_new_snapshots = number_of_shuffled_snapshots

            if snapshot_type == 'openpmd':
                import math
                import functools
                specified_number_of_new_snapshots = number_of_new_snapshots
                number_of_new_snapshots = functools.reduce(
                    math.gcd, [
                        snapshot.grid_dimension[0] for snapshot in
                        self.parameters.snapshot_directories_list
                    ], number_of_new_snapshots)
                if number_of_new_snapshots != specified_number_of_new_snapshots:
                    print(
                        f"[openPMD shuffling] Reduced the number of output snapshots to "
                        f"{number_of_new_snapshots} because of the dataset dimensions."
                    )
                del specified_number_of_new_snapshots

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

        if snapshot_type == 'numpy':
            self.__shuffle_numpy(number_of_new_snapshots, shuffle_dimensions,
                                 descriptor_save_path, save_name,
                                 target_save_path, permutations, file_ending)
        elif snapshot_type == 'openpmd':
            descriptor = self.DescriptorOrTarget(
                descriptor_save_path, lambda x: x.input_npy_directory,
                lambda x: x.input_npy_file, self.descriptor_calculator, ".in.",
                self.input_dimension)
            self.__shuffle_openpmd(descriptor, number_of_new_snapshots,
                                   shuffle_dimensions, save_name, permutations,
                                   file_ending)
            target = self.DescriptorOrTarget(target_save_path,
                                             lambda x: x.output_npy_directory,
                                             lambda x: x.output_npy_file,
                                             self.target_calculator, ".out.",
                                             self.output_dimension)
            self.__shuffle_openpmd(target, number_of_new_snapshots,
                                   shuffle_dimensions, save_name, permutations,
                                   file_ending)
        else:
            raise Exception("Unknown snapshot type: {}".format(snapshot_type))


        # Since no training will be done with this class, we should always
        # clear the data at the end.
        self.clear_data()
