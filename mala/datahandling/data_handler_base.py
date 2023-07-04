"""Base class for all data handling (loading, shuffling, etc.)."""
from abc import ABC
import os

import numpy as np

from mala.common.parameters import ParametersData, Parameters
from mala.common.parallelizer import printout
from mala.targets.target import Target
from mala.descriptors.descriptor import Descriptor
from mala.datahandling.snapshot import Snapshot


class DataHandlerBase(ABC):
    """
    Base class for all data handling (loading, shuffling, etc.).

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
        self.parameters: ParametersData = parameters.data
        self.use_horovod = parameters.use_horovod

        # Calculators used to parse data from compatible files.
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)
        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptor(parameters)

        # Dimensionalities of data.
        self.input_dimension = 0
        self.output_dimension = 0
        self.nr_snapshots = 0

    ##############################
    # Properties
    ##############################

    @property
    def input_dimension(self):
        """Feature dimension of input data."""
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, new_dimension):
        self._input_dimension = new_dimension

    @property
    def output_dimension(self):
        """Feature dimension of output data."""
        return self._output_dimension

    @output_dimension.setter
    def output_dimension(self, new_dimension):
        self._output_dimension = new_dimension

    ##############################
    # Public methods
    ##############################

    # Adding/Deleting data
    ########################

    def add_snapshot(self, input_file, input_directory,
                     output_file, output_directory,
                     add_snapshot_as,
                     output_units="1/(eV*A^3)", input_units="None",
                     calculation_output_file="", snapshot_type="numpy"):
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

        input_units : string
            Units of input data. See descriptor classes to see which units are
            supported.

        output_units : string
            Units of output data. See target classes to see which units are
            supported.

        calculation_output_file : string
            File with the output of the original snapshot calculation. This is
            only needed when testing multiple snapshots.

        add_snapshot_as : string
            Must be "tr", "va" or "te", the snapshot will be added to the
            snapshot list as training, validation or testing snapshot,
            respectively.

        snapshot_type : string
            Either "numpy" or "openpmd" based on what kind of files you
            want to operate on.
        """
        snapshot = Snapshot(input_file, input_directory,
                            output_file, output_directory,
                            add_snapshot_as,
                            input_units=input_units,
                            output_units=output_units,
                            calculation_output=calculation_output_file,
                            snapshot_type=snapshot_type)
        self.parameters.snapshot_directories_list.append(snapshot)

    def clear_data(self):
        """
        Reset the entire data pipeline.

        Useful when doing multiple investigations in the same python file.
        """
        self.parameters.snapshot_directories_list = []

    ##############################
    # Private methods
    ##############################

    # Loading data
    ######################

    def _check_snapshots(self, comm=None):
        """Check the snapshots for consistency."""
        self.nr_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots using a memorymap to see if there is consistency.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            printout("Checking descriptor file ", snapshot.input_npy_file,
                     "at", snapshot.input_npy_directory, min_verbosity=1)
            if snapshot.snapshot_type == "numpy":
                tmp_dimension = self.descriptor_calculator. \
                    read_dimensions_from_numpy_file(
                    os.path.join(snapshot.input_npy_directory,
                                 snapshot.input_npy_file))
            elif snapshot.snapshot_type == "openpmd":
                tmp_dimension = self.descriptor_calculator. \
                    read_dimensions_from_openpmd_file(
                    os.path.join(snapshot.input_npy_directory,
                                 snapshot.input_npy_file), comm=comm)
            else:
                raise Exception("Unknown snapshot file type.")

            # get the snapshot feature dimension - call it input dimension
            # for flexible grid sizes only this need be consistent
            tmp_input_dimension = tmp_dimension[-1]
            tmp_grid_dim = tmp_dimension[0:3]
            snapshot.grid_dimension = tmp_grid_dim
            snapshot.grid_size = int(np.prod(snapshot.grid_dimension))
            if firstsnapshot:
                self.input_dimension = tmp_input_dimension
            else:
                if self.input_dimension != tmp_input_dimension:
                    raise Exception("Invalid snapshot entered at ", snapshot.
                                    input_npy_file)
            ####################
            # Targets.
            ####################

            printout("Checking targets file ", snapshot.output_npy_file, "at",
                     snapshot.output_npy_directory, min_verbosity=1)
            if snapshot.snapshot_type == "numpy":
                tmp_dimension = self.target_calculator. \
                    read_dimensions_from_numpy_file(
                    os.path.join(snapshot.output_npy_directory,
                                 snapshot.output_npy_file))
            elif snapshot.snapshot_type == "openpmd":
                tmp_dimension = self.target_calculator. \
                    read_dimensions_from_openpmd_file(
                    os.path.join(snapshot.output_npy_directory,
                                 snapshot.output_npy_file), comm=comm)
            else:
                raise Exception("Unknown snapshot file type.")

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = tmp_dimension[-1]
            if firstsnapshot:
                self.output_dimension = tmp_output_dimension
            else:
                if self.output_dimension != tmp_output_dimension:
                    raise Exception("Invalid snapshot entered at ", snapshot.
                                    output_npy_file)

            if np.prod(tmp_dimension[0:3]) != snapshot.grid_size:
                raise Exception("Inconsistent snapshot data provided.")

            if firstsnapshot:
                firstsnapshot = False
