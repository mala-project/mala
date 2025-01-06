"""Mixes data between snapshots for improved lazy-loading training."""

import os

import numpy as np

from mala.common.parameters import (
    Parameters,
    DEFAULT_NP_DATA_DTYPE,
)
from mala.common.parallelizer import printout
from mala.common.physical_data import PhysicalData
from mala.datahandling.data_handler_base import DataHandlerBase
from mala.common.parallelizer import get_comm


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

    def __init__(
        self,
        parameters: Parameters,
        target_calculator=None,
        descriptor_calculator=None,
    ):
        super(DataShuffler, self).__init__(
            parameters,
            target_calculator=target_calculator,
            descriptor_calculator=descriptor_calculator,
        )
        if self.descriptor_calculator.parameters.descriptors_contain_xyz:
            printout(
                "Disabling XYZ-cutting from descriptor data for "
                "shuffling. If needed, please re-enable afterwards."
            )
            self.descriptor_calculator.parameters.descriptors_contain_xyz = (
                False
            )
        self._data_points_to_remove = None

    def add_snapshot(
        self,
        input_file,
        input_directory,
        output_file,
        output_directory,
        snapshot_type="numpy",
    ):
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
        super(DataShuffler, self).add_snapshot(
            input_file,
            input_directory,
            output_file,
            output_directory,
            add_snapshot_as="te",
            output_units="None",
            input_units="None",
            calculation_output_file="",
            snapshot_type=snapshot_type,
        )

    def __shuffle_numpy(
        self,
        number_of_new_snapshots,
        shuffle_dimensions,
        descriptor_save_path,
        save_name,
        target_save_path,
        permutations,
        file_ending,
    ):
        # Load the data (via memmap).
        descriptor_data = []
        target_data = []
        for idx, snapshot in enumerate(
            self.parameters.snapshot_directories_list
        ):
            # TODO: Use descriptor and target calculator for this.
            descriptor_data.append(
                np.load(
                    os.path.join(
                        snapshot.input_npy_directory, snapshot.input_npy_file
                    ),
                    mmap_mode="r",
                )
            )
            target_data.append(
                np.load(
                    os.path.join(
                        snapshot.output_npy_directory,
                        snapshot.output_npy_file,
                    ),
                    mmap_mode="r",
                )
            )

            # if the number of new snapshots is not a divisor of the grid size
            # then we have to trim the original snapshots to size
            # the indicies to be removed are selected at random
            if (
                self._data_points_to_remove is not None
                and np.sum(self._data_points_to_remove) > 0
            ):
                if self.parameters.shuffling_seed is not None:
                    np.random.seed(idx * self.parameters.shuffling_seed)
                ngrid = (
                    descriptor_data[idx].shape[0]
                    * descriptor_data[idx].shape[1]
                    * descriptor_data[idx].shape[2]
                )
                n_descriptor = descriptor_data[idx].shape[-1]
                n_target = target_data[idx].shape[-1]

                current_target = target_data[idx].reshape(-1, n_target)
                current_descriptor = descriptor_data[idx].reshape(
                    -1, n_descriptor
                )

                indices = np.random.choice(
                    ngrid,
                    size=ngrid - self._data_points_to_remove[idx],
                )

                descriptor_data[idx] = current_descriptor[indices]
                target_data[idx] = current_target[indices]

        # Do the actual shuffling.
        target_name_openpmd = os.path.join(
            target_save_path, save_name.replace("*", "%T")
        )
        descriptor_name_openpmd = os.path.join(
            descriptor_save_path, save_name.replace("*", "%T")
        )
        for i in range(0, number_of_new_snapshots):
            new_descriptors = np.zeros(
                (int(np.prod(shuffle_dimensions)), self.input_dimension),
                dtype=DEFAULT_NP_DATA_DTYPE,
            )
            new_targets = np.zeros(
                (int(np.prod(shuffle_dimensions)), self.output_dimension),
                dtype=DEFAULT_NP_DATA_DTYPE,
            )
            last_start = 0
            descriptor_name = os.path.join(
                descriptor_save_path, save_name.replace("*", str(i))
            )
            target_name = os.path.join(
                target_save_path, save_name.replace("*", str(i))
            )

            # Each new snapshot gets an number_of_new_snapshots-th from each
            # snapshot.
            for j in range(0, self.nr_snapshots):
                current_grid_size = self.parameters.snapshot_directories_list[
                    j
                ].grid_size
                current_chunk = int(
                    current_grid_size / number_of_new_snapshots
                )
                new_descriptors[
                    last_start : current_chunk + last_start
                ] = descriptor_data[j].reshape(-1, self.input_dimension)[
                    i * current_chunk : (i + 1) * current_chunk, :
                ]
                new_targets[
                    last_start : current_chunk + last_start
                ] = target_data[j].reshape(-1, self.output_dimension)[
                    i * current_chunk : (i + 1) * current_chunk, :
                ]

                last_start += current_chunk

            # Randomize and save to disk.
            new_descriptors = new_descriptors[permutations[i]]
            new_targets = new_targets[permutations[i]]
            new_descriptors = new_descriptors.reshape(
                [
                    shuffle_dimensions[0],
                    shuffle_dimensions[1],
                    shuffle_dimensions[2],
                    self.input_dimension,
                ]
            )
            new_targets = new_targets.reshape(
                [
                    shuffle_dimensions[0],
                    shuffle_dimensions[1],
                    shuffle_dimensions[2],
                    self.output_dimension,
                ]
            )
            if file_ending == "npy":
                self.descriptor_calculator.write_to_numpy_file(
                    descriptor_name + ".in.npy", new_descriptors
                )
                self.target_calculator.write_to_numpy_file(
                    target_name + ".out.npy", new_targets
                )
            else:
                # We check above that in the non-numpy case, OpenPMD will work.
                self.descriptor_calculator.grid_dimensions = list(
                    shuffle_dimensions
                )
                self.target_calculator.grid_dimensions = list(
                    shuffle_dimensions
                )
                self.descriptor_calculator.write_to_openpmd_file(
                    descriptor_name_openpmd + ".in." + file_ending,
                    new_descriptors,
                    additional_attributes={
                        "global_shuffling_seed": self.parameters.shuffling_seed,
                        "local_shuffling_seed": i
                        * self.parameters.shuffling_seed,
                    },
                    internal_iteration_number=i,
                )
                self.target_calculator.write_to_openpmd_file(
                    target_name_openpmd + ".out." + file_ending,
                    array=new_targets,
                    additional_attributes={
                        "global_shuffling_seed": self.parameters.shuffling_seed,
                        "local_shuffling_seed": i
                        * self.parameters.shuffling_seed,
                    },
                    internal_iteration_number=i,
                )

    # The function __shuffle_openpmd can be used to shuffle descriptor data and
    # target data.
    # It will be executed one after another for both of them.
    # Use this class to parameterize which of both should be shuffled.
    class __DescriptorOrTarget:
        def __init__(
            self,
            save_path,
            npy_directory,
            npy_file,
            calculator,
            name_infix,
            dimension,
        ):
            self.save_path = save_path
            self.npy_directory = npy_directory
            self.npy_file = npy_file
            self.calculator = calculator
            self.name_infix = name_infix
            self.dimension = dimension

    class __MockedMPIComm:
        def __init__(self):
            self.rank = 0
            self.size = 1

    # Takes the `extent` of an n-dimensional grid, the coordinates of a start
    # point `x` and of an end point `y`.
    # Interpreting the grid as a row-major n-dimensional array yields a
    # contiguous slice from `x` to `y`.
    # !!! Both `x` and `y` are **inclusive** ends as the semantics of what would
    # be the next exclusive upper limit are unnecessarily complex in
    # n dimensions, especially since the function recurses over the number of
    # dimensions. !!!
    # This slice is not in general an n-dimensional cuboid within the grid, but
    # may be a non-convex form composed of multiple cuboid chunks.
    #
    # This function returns a list of cuboid sub-chunks of the n-dimensional
    # grid such that:
    #
    # 1. The union of those chunks is equal to the contiguous slice spanned
    #    between `x` and `y`, both ends inclusive.
    # 2. The chunks do not overlap.
    # 3. Each single chunk is a contiguous slice within the row-major grid.
    # 4. The chunks are sorted in ascending order of their position within the
    #    row-major grid.
    #
    # Row-major within the context of this function means that the indexes go
    # from the slowest-running dimension at the left end to the fastest-running
    # dimension at the right end.
    #
    # The output is a list of chunks as defined in openPMD, i.e. a start offset
    # and the extent, counted from this offset (not from the grid origin!).
    #
    # Example: From the below 2D grid with extent [8, 10], the slice marked by
    # the X-es should be loaded. The slice is defined by its first item
    # at [2, 3] and its last item at [6, 6].
    # The result would be a list of three chunks:
    #
    # 1. ([2, 3], [1,  7])
    # 2. ([3, 0], [3, 10])
    # 3. ([6, 0], [1,  7])
    #
    # OOOOOOOOOO
    # OOOOOOOOOO
    # OOOXXXXXXX
    # XXXXXXXXXX
    # XXXXXXXXXX
    # XXXXXXXXXX
    # XXXXXXXOOO
    # OOOOOOOOOO

    def __contiguous_slice_within_ndim_grid_as_blocks(
        extent: list[int], x: list[int], y: list[int]
    ):
        # Used for converting a block defined by inclusive lower and upper
        # coordinate to a chunk as defined by openPMD.
        # The openPMD extent is defined by (to-from)+1 (to make up for the
        # inclusive upper end).
        def get_extent(from_, to_):
            res = [upper + 1 - lower for (lower, upper) in zip(from_, to_)]
            if any(x < 1 for x in res):
                raise RuntimeError(
                    f"Cannot compute block extent from {from_} to {to_}."
                )
            return res

        if len(extent) != len(x):
            raise RuntimeError(
                f"__shuffle_openpmd: Internal indexing error extent={extent} and x={x} must have the same length. This is a bug."
            )
        if len(extent) != len(y):
            raise RuntimeError(
                f"__shuffle_openpmd: Internal indexing error extent={extent} and y={y} must have the same length. This is a bug."
            )

        # Recursive bottom cases
        # 1. No items left
        if y < x:
            return []
        # 2. No distinct dimensions left
        elif x[0:-1] == y[0:-1]:
            return [(x.copy(), get_extent(x, y))]

        # Take the frontside and backside one-dimensional slices with extent
        # defined by the last dimension, process the remainder recursively.

        # The offset of the front slice is equal to x.
        front_slice = (x.copy(), [1 for _ in range(len(x))])
        # The chunk extent in the last dimension is the distance from the x
        # coordinate to the grid extent in the last dimension.
        # As the slice is one-dimensional, the extent is 1 in all other
        # dimensions.
        front_slice[1][-1] = extent[-1] - x[-1]

        # Similar for the back slice.
        # The offset of the back slice is 0 in the last dimension, equal to y
        # in all other dimensions.
        back_slice = (y.copy(), [1 for _ in range(len(y))])
        back_slice[0][-1] = 0
        # The extent is equal to the coordinate of y+1 (to convert inclusive to
        # exclusive upper end) in the last dimension, 1 otherwise.
        back_slice[1][-1] = y[-1] + 1

        # Strip the last (i.e. fast-running) dimension for a recursive call with
        # one dimensionality reduced.
        recursive_x = x[0:-1]
        recursive_y = y[0:-1]

        # Since the first and last row are dealt with above, those rows are now
        # stripped from the second-to-last dimension for the recursive call
        # (in which they will be the last dimension)
        recursive_x[-1] += 1
        recursive_y[-1] -= 1

        rec_res = DataShuffler.__contiguous_slice_within_ndim_grid_as_blocks(
            extent[0:-1], recursive_x, recursive_y
        )

        # Add the last dimension again. The last dimension is covered from start
        # to end since the leftovers have been dealt with by the front and back
        # slice.
        rec_res = [(xx + [0], yy + [extent[-1]]) for (xx, yy) in rec_res]

        return [front_slice] + rec_res + [back_slice]

    # Interpreting `ndim_extent` as the extents of an n-dimensional grid,
    # returns the n-dimensional coordinate of the `idx`th item in the row-major
    # representation of the grid.
    def __resolve_flattened_index_into_ndim(idx: int, ndim_extent: list[int]):
        if not ndim_extent:
            raise RuntimeError("Cannot index into a zero-dimensional array.")
        strides = []
        current_stride = 1
        for ext in reversed(ndim_extent):
            strides = [current_stride] + strides
            # sic!, the last stride is ignored, as it's just the entire grid
            # extent
            current_stride *= ext

        def worker(inner_idx, inner_strides):
            if not inner_strides:
                if inner_idx != 0:
                    raise RuntimeError(
                        "This cannot happen. There is bug somewhere.")
                else:
                    return []
            div, mod = divmod(inner_idx, inner_strides[0])
            return [div] + worker(mod, inner_strides[1:])

        return worker(idx, strides)

    # Load a chunk from the openPMD `record` into the buffer at `arr`.
    # The indexes `offset` and `extent` are scalar 1-dimensional coordinates,
    # and apply to the n-dimensional record by reinterpreting (reshaped) it as
    # a one-dimensional array.
    # The function deals internally with splitting the 1-dimensional slice into
    # a sequence of n-dimensional block load operations.
    def __load_chunk_1D(mesh, arr, offset, extent):
        start_idx = DataShuffler.__resolve_flattened_index_into_ndim(
            offset, mesh.shape
        )
        # Inclusive upper end. See the documentation in
        # __contiguous_slice_within_ndim_grid_as_blocks() for the reason why
        # we work with inclusive upper ends.
        end_idx = DataShuffler.__resolve_flattened_index_into_ndim(
            offset + extent - 1, mesh.shape
        )
        blocks_to_load = (
            DataShuffler.__contiguous_slice_within_ndim_grid_as_blocks(
                mesh.shape, start_idx, end_idx
            )
        )
        # print(f"\n\nLOADING {offset}\t+{extent}\tFROM {mesh.shape}")
        current_offset = 0  # offset within arr
        for nd_offset, nd_extent in blocks_to_load:
            flat_extent = np.prod(nd_extent)
            # print(
            #     f"\t{nd_offset}\t-{nd_extent}\t->[{current_offset}:{current_offset + flat_extent}]"
            # )
            mesh.load_chunk(
                arr[current_offset : current_offset + flat_extent],
                nd_offset,
                nd_extent,
            )
            current_offset += flat_extent

    def __shuffle_openpmd(
        self,
        dot: __DescriptorOrTarget,
        number_of_new_snapshots,
        shuffle_dimensions,
        save_name,
        permutations,
        file_ending,
    ):
        import openpmd_api as io

        if self.parameters._configuration["mpi"]:
            comm = get_comm()
        else:
            comm = self.__MockedMPIComm()

        import math

        items_per_process = math.ceil(number_of_new_snapshots / comm.size)
        my_items_start = comm.rank * items_per_process
        my_items_end = min(
            (comm.rank + 1) * items_per_process, number_of_new_snapshots
        )
        my_items_count = my_items_end - my_items_start

        if self.parameters._configuration["mpi"]:
            # imagine we have 20 new snapshots to create, but 100 ranks
            # it's sufficient to let only the first 20 ranks participate in the
            # following code
            num_of_participating_ranks = math.ceil(
                number_of_new_snapshots / items_per_process
            )
            color = comm.rank < num_of_participating_ranks
            comm = comm.Split(color=int(color), key=comm.rank)
            if not color:
                return

        # Load the data
        input_series_list = []
        for idx, snapshot in enumerate(
            self.parameters.snapshot_directories_list
        ):
            # TODO: Use descriptor and target calculator for this.
            if isinstance(comm, self.__MockedMPIComm):
                input_series_list.append(
                    io.Series(
                        os.path.join(
                            dot.npy_directory(snapshot),
                            dot.npy_file(snapshot),
                        ),
                        io.Access.read_only,
                    )
                )
            else:
                input_series_list.append(
                    io.Series(
                        os.path.join(
                            dot.npy_directory(snapshot),
                            dot.npy_file(snapshot),
                        ),
                        io.Access.read_only,
                        comm,
                    )
                )

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
        def from_chunk_i(i, n, dset):
            if isinstance(dset, io.Dataset):
                dset = dset.extent
            flat_extent = np.prod(dset)
            one_slice_extent = flat_extent // n
            return i * one_slice_extent, one_slice_extent

        import json

        # Do the actual shuffling.
        name_prefix = os.path.join(dot.save_path, save_name.replace("*", "%T"))
        for i in range(my_items_start, my_items_end):
            # We check above that in the non-numpy case, OpenPMD will work.
            dot.calculator.grid_dimensions = list(shuffle_dimensions)
            # do NOT open with MPI
            shuffled_snapshot_series = io.Series(
                name_prefix + dot.name_infix + file_ending,
                io.Access.create,
                options=json.dumps(
                    self.parameters._configuration["openpmd_configuration"]
                ),
            )
            dot.calculator.write_to_openpmd_file(
                shuffled_snapshot_series,
                PhysicalData.SkipArrayWriting(dataset, feature_size),
                additional_attributes={
                    "global_shuffling_seed": self.parameters.shuffling_seed,
                    "local_shuffling_seed": i * self.parameters.shuffling_seed,
                },
                internal_iteration_number=i,
            )
            mesh_out = shuffled_snapshot_series.write_iterations()[i].meshes[
                dot.calculator.data_name
            ]
            new_array = np.zeros(
                (dot.dimension, int(np.prod(shuffle_dimensions))),
                dtype=dataset.dtype,
            )

            # Need to add to these in the loop as the single chunks might have
            # different sizes
            to_chunk_offset, to_chunk_extent = 0, 0
            for j in range(0, self.nr_snapshots):
                extent_in = self.parameters.snapshot_directories_list[
                    j
                ].grid_dimension
                if len(input_series_list[j].iterations) != 1:
                    raise Exception(
                        "Input Series '{}' has {} iterations (needs exactly one).".format(
                            input_series_list[j].name,
                            len(input_series_list[j].iterations),
                        )
                    )
                for iteration in input_series_list[j].read_iterations():
                    mesh_in = iteration.meshes[dot.calculator.data_name]
                    break

                # Note that the semantics of from_chunk_extent and
                # to_chunk_extent are not the same.
                # from_chunk_extent describes the size of the chunk, as is usual
                # in openPMD, to_chunk_extent describes the upper coordinate of
                # the slice, as is usual in Python.
                from_chunk_offset, from_chunk_extent = from_chunk_i(
                    i, number_of_new_snapshots, extent_in
                )
                to_chunk_offset = to_chunk_extent
                to_chunk_extent = to_chunk_offset + from_chunk_extent
                for dimension in range(len(mesh_in)):
                    DataShuffler.__load_chunk_1D(
                        mesh_in[str(dimension)],
                        new_array[dimension, to_chunk_offset:to_chunk_extent],
                        from_chunk_offset,
                        from_chunk_extent,
                    )
                mesh_in.series_flush()

            for k in range(feature_size):
                rc = mesh_out[str(k)]
                rc[:, :, :] = new_array[k, :][permutations[i]].reshape(
                    shuffle_dimensions
                )
            shuffled_snapshot_series.close()

        # Ensure consistent parallel destruction
        # Closing a series is a collective operation
        for series in input_series_list:
            series.close()

    def shuffle_snapshots(
        self,
        complete_save_path=None,
        descriptor_save_path=None,
        target_save_path=None,
        save_name="mala_shuffled_snapshot*",
        number_of_shuffled_snapshots=None,
    ):
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
                    raise Exception(
                        "Invalid file ending selected: " + file_ending
                    )
        else:
            file_ending = "npy"

        if self.parameters._configuration["mpi"]:
            self._check_snapshots(comm=get_comm())
        else:
            self._check_snapshots()

        snapshot_types = {
            snapshot.snapshot_type
            for snapshot in self.parameters.snapshot_directories_list
        }
        if len(snapshot_types) > 1:
            raise Exception(
                "[data_shuffler] Can only deal with one type of input snapshot"
                + " at once (openPMD or numpy)."
            )
        snapshot_type = snapshot_types.pop()
        del snapshot_types

        # Set the defaults, these may be changed below as needed.
        snapshot_size_list = np.array(
            [
                snapshot.grid_size
                for snapshot in self.parameters.snapshot_directories_list
            ]
        )
        number_of_data_points = np.sum(snapshot_size_list)
        self._data_points_to_remove = None
        if number_of_shuffled_snapshots is None:
            number_of_shuffled_snapshots = self.nr_snapshots

        shuffled_gridsizes = snapshot_size_list // number_of_shuffled_snapshots

        if np.any(
            np.array(snapshot_size_list)
            - (
                (np.array(snapshot_size_list) // number_of_shuffled_snapshots)
                * number_of_shuffled_snapshots
            )
            > 0
        ):
            number_of_data_points = int(
                np.sum(shuffled_gridsizes) * number_of_shuffled_snapshots
            )

        self._data_points_to_remove = []
        for i in range(0, self.nr_snapshots):
            self._data_points_to_remove.append(
                snapshot_size_list[i]
                - shuffled_gridsizes[i] * number_of_shuffled_snapshots
            )
        tot_points_missing = sum(self._data_points_to_remove)

        if tot_points_missing > 0:
            printout(
                "Warning: number of requested snapshots is not a divisor of",
                "the original grid sizes.\n",
                f"{tot_points_missing} / {number_of_data_points} data points",
                "will be left out of the shuffled snapshots.",
            )

        shuffle_dimensions = [
            int(number_of_data_points / number_of_shuffled_snapshots),
            1,
            1,
        ]

        printout(
            "Data shuffler will generate",
            number_of_shuffled_snapshots,
            "new snapshots.",
        )
        printout("Shuffled snapshot dimension will be ", shuffle_dimensions)

        # Prepare permutations.
        permutations = []
        seeds = []
        for i in range(0, number_of_shuffled_snapshots):
            # This makes the shuffling deterministic, if specified by the user.
            if self.parameters.shuffling_seed is not None:
                np.random.seed(i * self.parameters.shuffling_seed)
            permutations.append(
                np.random.permutation(int(np.prod(shuffle_dimensions)))
            )

        if snapshot_type == "numpy":
            self.__shuffle_numpy(
                number_of_shuffled_snapshots,
                shuffle_dimensions,
                descriptor_save_path,
                save_name,
                target_save_path,
                permutations,
                file_ending,
            )
        elif snapshot_type == "openpmd":
            descriptor = self.__DescriptorOrTarget(
                descriptor_save_path,
                lambda x: x.input_npy_directory,
                lambda x: x.input_npy_file,
                self.descriptor_calculator,
                ".in.",
                self.input_dimension,
            )
            self.__shuffle_openpmd(
                descriptor,
                number_of_shuffled_snapshots,
                shuffle_dimensions,
                save_name,
                permutations,
                file_ending,
            )
            target = self.__DescriptorOrTarget(
                target_save_path,
                lambda x: x.output_npy_directory,
                lambda x: x.output_npy_file,
                self.target_calculator,
                ".out.",
                self.output_dimension,
            )
            self.__shuffle_openpmd(
                target,
                number_of_shuffled_snapshots,
                shuffle_dimensions,
                save_name,
                permutations,
                file_ending,
            )
        else:
            raise Exception("Unknown snapshot type: {}".format(snapshot_type))

        # Since no training will be done with this class, we should always
        # clear the data at the end.
        self.clear_data()
