"""DataConverter class for converting snapshots into numpy arrays."""
import os

import numpy as np

from mala.common.parallelizer import printout, get_rank
from mala.common.parameters import ParametersData
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target


class DataConverter:
    """
    Converts raw snapshots (direct DFT outputs) into numpy arrays for MALA.

    These snapshots can be e.g. Quantum Espresso results.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        The parameters object used for creating this instance.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        The descriptor calculator used for parsing/converting fingerprint
        data. If None, the descriptor calculator will be created by this
        object using the parameters provided. Default: None

    target_calculator : mala.targets.target.Target
        Target calculator used for parsing/converting target data. If None,
        the target calculator will be created by this object using the
        parameters provided. Default: None

    Attributes
    ----------
    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Descriptor calculator used for parsing/converting fingerprint data.

    target_calculator : mala.targets.target.Target
        Target calculator used for parsing/converting target data.
    """

    def __init__(self, parameters, descriptor_calculator=None,
                 target_calculator=None):
        self.parameters: ParametersData = parameters.data
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptor(parameters)

        self.__snapshots_to_convert = []
        self.__snapshot_description = []
        self.__snapshot_units = []

    def add_snapshot_qeout_cube(self, qe_out_file, qe_out_directory,
                                cube_naming_scheme, cube_directory,
                                input_units=None, output_units=None):
        """
        Add a Quantum Espresso snapshot to the list of conversion list.

        Please note that a Quantum Espresso snapshot consists of:

            - a Quantum Espresso output file for a self-consistent calculation.
            - multiple .cube files containing the LDOS

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for this snapshot.

        qe_out_directory : string
            Path to Quantum Espresso output file for this snapshot.

        cube_naming_scheme : string
            Naming scheme for the LDOS .cube files.

        cube_directory : string
            Directory containing the LDOS .cube files.

        input_units : string
            Unit of the input data.

        output_units : string
            Unit of the output data.
        """
        self.__snapshots_to_convert.append([qe_out_file, qe_out_directory,
                                            cube_naming_scheme,
                                            cube_directory])
        self.__snapshot_description.append(["qe.out", ".cube"])
        self.__snapshot_units.append([input_units, output_units])

    def convert_single_snapshot(self, snapshot_number,
                                input_path=None,
                                output_path=None,
                                use_memmap=None,
                                return_data=False):
        """
        Convert single snapshot from the conversion lists.

        Returns the preprocessed data as numpy array, which might be
        beneficial during testing.

        Parameters
        ----------
        snapshot_number : integer
            Position of the desired snapshot in the snapshot list.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS.
            If run in MPI parallel mode, such a file MUST be provided.

        input_path : string
            If not None, inputs will be saved in this file.

        output_path : string
            If not None, outputs will be saved in this file.

        return_data : bool
            If True, inputs and outputs will be returned directly.

        Returns
        -------
        inputs : numpy.array , optional
            Numpy array containing the preprocessed inputs.

        outputs : numpy.array , optional
            Numpy array containing the preprocessed outputs.
        """
        snapshot = self.__snapshots_to_convert[snapshot_number]
        description = self.__snapshot_description[snapshot_number]
        original_units = self.__snapshot_units[snapshot_number]

        # Parse and/or calculate the input descriptors.
        if description[0] == "qe.out":
            if original_units[0] is None:
                tmp_input, local_size = self.descriptor_calculator. \
                    calculate_from_qe_out(snapshot[0], snapshot[1])
            else:
                tmp_input, local_size = self.descriptor_calculator. \
                    calculate_from_qe_out(snapshot[0], snapshot[1],
                                          units=original_units[0])
            if self.parameters._configuration["mpi"]:
                tmp_input = self.descriptor_calculator.gather_descriptors(tmp_input)

            # Cut the xyz information if requested by the user.
            if get_rank() == 0:
                if self.descriptor_calculator.descriptors_contain_xyz is False:
                    tmp_input = tmp_input[:, :, :, 3:]

        else:
            raise Exception("Unknown file extension, cannot convert target")

        # Save data and delete, if not requested otherwise.
        if get_rank() == 0:
            if input_path is not None:
                np.save(input_path, tmp_input)

        if not return_data:
            del tmp_input

        # Parse and/or calculate the output descriptors.
        if description[1] == ".cube":

            # If no units are provided we just assume standard units.
            if original_units[1] is None:
                tmp_output = self.target_calculator.read_from_cube(
                    snapshot[2], snapshot[3], use_memmap=use_memmap)
            else:
                tmp_output = self.target_calculator. \
                    read_from_cube(snapshot[2], snapshot[3], units=
                original_units[1], use_memmap=use_memmap)
        else:
            raise Exception(
                "Unknown file extension, cannot convert target"
                "data.")

        if get_rank() == 0:
            if output_path is not None:
                np.save(output_path, tmp_output)

        if return_data:
            return tmp_input, tmp_output

    def convert_snapshots(self, save_path="./",
                          naming_scheme="ELEM_snapshot*", starts_at=0,
                          file_based_communication=False):
        """
        Convert the snapshots in the list to numpy arrays.

        These can then be used by MALA.

        Parameters
        ----------
        save_path : string
            Directory in which the snapshots will be saved.

        naming_scheme : string
            String detailing the naming scheme for the snapshots. * symbols
            will be replaced with the snapshot number.

        starts_at : int
            Number of the first snapshot generated using this approach.
            Default is 0, but may be set to any integer. This is to ensure
            consistency in naming when converting e.g. only a certain portion
            of all available snapshots. If set to e.g. 4,
            the first snapshot generated will be called snapshot4.

        file_based_communication : bool
            If True, the LDOS will be gathered using a file based mechanism.
            This is drastically less performant then using MPI, but may be
            necessary when memory is scarce. Default is False, i.e., the faster
            MPI version will be used.
        """
        for i in range(0, len(self.__snapshots_to_convert)):
            snapshot_number = i + starts_at
            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(snapshot_number))

            # A memory mapped file is used as buffer for distributed cases.
            memmap = None
            if self.parameters._configuration["mpi"] and \
                    file_based_communication:
                memmap = os.path.join(save_path, snapshot_name +
                                      ".out.npy_temp")

            self.convert_single_snapshot(i, input_path=os.path.join(save_path,
                                         snapshot_name+".in.npy"),
                                         output_path=os.path.join(save_path,
                                         snapshot_name+".out.npy"),
                                         use_memmap=memmap)
            printout("Saved snapshot", snapshot_number, "at ", save_path,
                     min_verbosity=0)

            if get_rank() == 0:
                if self.parameters._configuration["mpi"] \
                        and file_based_communication:
                    os.remove(memmap)
