"""Base class for all data handling (loading, shuffling, etc.)."""

from abc import ABC
import os
import tempfile

import numpy as np

from mala.common.parameters import ParametersData, Parameters
from mala.common.parallelizer import printout, get_rank, barrier
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

    Attributes
    ----------
    descriptor_calculator
        Used to do unit conversion on input data.

    nr_snapshots : int
        Number of snapshots loaded.

    parameters : mala.common.parameters.ParametersData
        MALA data handling parameters.

    target_calculator
        Used to do unit conversion on output data.
    """

    def __init__(
        self,
        parameters: Parameters,
        target_calculator=None,
        descriptor_calculator=None,
    ):
        self.parameters: ParametersData = parameters.data
        self._use_ddp = parameters.use_ddp

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

    def add_snapshot(
        self,
        input_file,
        input_directory,
        output_file,
        output_directory,
        add_snapshot_as,
        output_units="1/(eV*A^3)",
        input_units="None",
        calculation_output_file="",
        snapshot_type=None,
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
        # Try to guess snapshot type if no information was provided.
        if snapshot_type is None:
            input_file_ending = input_file.split(".")[-1]
            output_file_ending = output_file.split(".")[-1]

            if input_file_ending == "npy" and output_file_ending == "npy":
                snapshot_type = "numpy"

            elif input_file_ending == "json" and output_file_ending == "npy":
                snapshot_type = "json+numpy"
            else:
                import openpmd_api as io

                if (
                    input_file_ending in io.file_extensions
                    and output_file_ending in io.file_extensions
                ):
                    snapshot_type = "openpmd"
                if (
                    input_file_ending == "json"
                    and output_file_ending in io.file_extensions
                ):
                    snapshot_type = "json+openpmd"

        snapshot = Snapshot(
            input_file,
            input_directory,
            output_file,
            output_directory,
            add_snapshot_as,
            input_units=input_units,
            output_units=output_units,
            calculation_output=calculation_output_file,
            snapshot_type=snapshot_type,
        )
        self.parameters.snapshot_directories_list.append(snapshot)

    def clear_data(self):
        """
        Reset the entire data pipeline.

        Useful when doing multiple investigations in the same python file.
        """
        self.parameters.snapshot_directories_list = []

    def delete_temporary_inputs(self):
        """
        Delete temporary data files.

        These may have been created during a training or testing process
        when using atomic positions for on-the-fly calculation of descriptors
        rather than precomputed data files.
        """
        if get_rank() == 0:
            for snapshot in self.parameters.snapshot_directories_list:
                if snapshot.temporary_input_file is not None:
                    if os.path.isfile(snapshot.temporary_input_file):
                        os.remove(snapshot.temporary_input_file)
        barrier()

    ##############################
    # Private methods
    ##############################

    # Loading data
    ######################

    def _check_snapshots(self, comm=None):
        """
        Check the snapshots for consistency.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            MPI communicator used for communication between ranks.
        """
        self.nr_snapshots = len(self.parameters.snapshot_directories_list)

        # Read the snapshots using a memorymap to see if there is consistency.
        firstsnapshot = True
        for snapshot in self.parameters.snapshot_directories_list:
            ####################
            # Descriptors.
            ####################

            printout(
                "Checking descriptor file ",
                snapshot.input_npy_file,
                "at",
                snapshot.input_npy_directory,
                min_verbosity=1,
            )
            if snapshot.snapshot_type == "numpy":
                tmp_dimension = (
                    self.descriptor_calculator.read_dimensions_from_numpy_file(
                        os.path.join(
                            snapshot.input_npy_directory,
                            snapshot.input_npy_file,
                        )
                    )
                )
            elif snapshot.snapshot_type == "openpmd":
                tmp_dimension = self.descriptor_calculator.read_dimensions_from_openpmd_file(
                    os.path.join(
                        snapshot.input_npy_directory, snapshot.input_npy_file
                    ),
                    comm=comm,
                )
            elif (
                snapshot.snapshot_type == "json+numpy"
                or snapshot.snapshot_type == "json+openpmd"
            ):
                tmp_dimension = (
                    self.descriptor_calculator.read_dimensions_from_json(
                        os.path.join(
                            snapshot.input_npy_directory,
                            snapshot.input_npy_file,
                        )
                    )
                )
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
                    raise Exception(
                        "Invalid snapshot entered at ",
                        snapshot.input_npy_file,
                    )
            ####################
            # Targets.
            ####################

            printout(
                "Checking targets file ",
                snapshot.output_npy_file,
                "at",
                snapshot.output_npy_directory,
                min_verbosity=1,
            )
            if (
                snapshot.snapshot_type == "numpy"
                or snapshot.snapshot_type == "json+numpy"
            ):
                tmp_dimension = (
                    self.target_calculator.read_dimensions_from_numpy_file(
                        os.path.join(
                            snapshot.output_npy_directory,
                            snapshot.output_npy_file,
                        )
                    )
                )
            elif (
                snapshot.snapshot_type == "openpmd"
                or snapshot.snapshot_type == "json+openpmd"
            ):
                tmp_dimension = (
                    self.target_calculator.read_dimensions_from_openpmd_file(
                        os.path.join(
                            snapshot.output_npy_directory,
                            snapshot.output_npy_file,
                        ),
                        comm=comm,
                    )
                )
            else:
                raise Exception("Unknown snapshot file type.")

            # The first snapshot determines the data size to be used.
            # We need to make sure that snapshot size is consistent.
            tmp_output_dimension = tmp_dimension[-1]
            if firstsnapshot:
                self.output_dimension = tmp_output_dimension
            else:
                if self.output_dimension != tmp_output_dimension:
                    raise Exception(
                        "Invalid snapshot entered at ",
                        snapshot.output_npy_file,
                    )

            if np.prod(tmp_dimension[0:3]) != snapshot.grid_size:
                raise Exception("Inconsistent snapshot data provided.")

            if firstsnapshot:
                firstsnapshot = False

    def _calculate_temporary_inputs(self, snapshot: Snapshot):
        """
        Calculate temporary input files.

        If a MALA generated JSON file is used as input data, then the
        descriptors for this snapshot have to be calculated here.
        If the descriptor has already been calculated, then no computation
        is performed here. This can happen during interrupted and resumed
        training.

        Parameters
        ----------
        snapshot : mala.datahandling.snapshot.Snapshot
            Snapshot for which to compute temporary inputs.
        """
        if snapshot.temporary_input_file is not None:
            if not os.path.isfile(snapshot.temporary_input_file):
                snapshot.temporary_input_file = None

        if snapshot.temporary_input_file is None:
            snapshot.temporary_input_file = tempfile.NamedTemporaryFile(
                delete=False,
                prefix=snapshot.input_npy_file.split(".")[0],
                suffix=".in.npy",
                dir=snapshot.input_npy_directory,
            ).name
            tmp, grid = self.descriptor_calculator.calculate_from_json(
                os.path.join(
                    snapshot.input_npy_directory,
                    snapshot.input_npy_file,
                )
            )
            if self.parameters._configuration["mpi"]:
                tmp = self.descriptor_calculator.gather_descriptors(tmp)

            if get_rank() == 0:
                np.save(snapshot.temporary_input_file, tmp)
            barrier()
