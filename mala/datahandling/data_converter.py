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

        if parameters.descriptors.use_z_splitting:
            parameters.descriptors.use_z_splitting = False
            printout("Disabling z-splitting for preprocessing.",
                     min_verbosity=0)

        self.__snapshots_to_convert = []
        self.__snapshot_description = []
        self.__snapshot_units = []

    def add_snapshot_qeout_cube(self, qe_out_file, cube_files_scheme,
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

        cube_files_scheme : string
            Naming scheme for the LDOS .cube files.

        input_units : string
            Unit of the input data.

        output_units : string
            Unit of the output data.
        """
        self.__snapshots_to_convert.append({"input": [qe_out_file],
                                            "output": [cube_files_scheme]})
        self.__snapshot_description.append({"input": "qe.out",
                                            "output": ".cube"})
        self.__snapshot_units.append({"input": input_units,
                                      "output": output_units})

    def add_snapshot_qeout(self, qe_out_file, input_units=None):
        """
        Add a Quantum Espresso snapshot to the list of conversion list.

        This snapshot only contains a QE.out file, i.e., only the SNAP
        descriptors will be calculated. Useful if the output data has already
        been processed.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for this snapshot.

        input_units : string
            Unit of the input data.
        """
        self.__snapshots_to_convert.append({"input": [qe_out_file],
                                            "output": None})
        self.__snapshot_description.append({"input": "qe.out",
                                            "output": None})
        self.__snapshot_units.append({"input": input_units,
                                      "output": None})

    def add_snapshot_cube(self, cube_naming_scheme, output_units=None):
        """
        Add a Quantum Espresso snapshot to the list of conversion list.

        This snapshot only contains the cube files for the output
        quantity (LDOS/density), so only the output quantity will be
        calculated. Useful if the input data has already been processed.

        Parameters
        ----------
        cube_naming_scheme : string
            Naming scheme for the LDOS .cube files.

        output_units : string
            Unit of the output data.
        """
        self.__snapshots_to_convert.append({"input": None,
                                            "output": [cube_naming_scheme]})
        self.__snapshot_description.append({"input": None,
                                            "output": ".cube"})
        self.__snapshot_units.append({"input": None,
                                      "output": output_units})

    def convert_single_snapshot(self, snapshot_number,
                                descriptor_calculation_kwargs,
                                target_calculator_kwargs,
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

        target_calculator_kwargs : dict
            Dictionary with additional keyword arguments for the calculation
            or parsing of the target quantities.

        descriptor_calculation_kwargs : dict
            Dictionary with additional keyword arguments for the calculation
            or parsing of the descriptor quantities.

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
        if description["input"] == "qe.out":
            descriptor_calculation_kwargs["units"] = original_units["input"]
            tmp_input, local_size = self.descriptor_calculator. \
                calculate_from_qe_out(snapshot["input"][0],
                                      **descriptor_calculation_kwargs)
            if self.parameters._configuration["mpi"]:
                tmp_input = self.descriptor_calculator.gather_descriptors(tmp_input)

            # Cut the xyz information if requested by the user.
            if get_rank() == 0:
                if self.descriptor_calculator.descriptors_contain_xyz is False:
                    tmp_input = tmp_input[:, :, :, 3:]

        elif description["input"] is None:
            # In this case, only the output is processed.
            pass

        else:
            raise Exception("Unknown file extension, cannot convert target")

        if description["input"] is not None:
            # Save data and delete, if not requested otherwise.
            if get_rank() == 0:
                if input_path is not None:
                    np.save(input_path, tmp_input)

            if not return_data:
                del tmp_input

        # Parse and/or calculate the output descriptors.
        if description["output"] == ".cube":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = use_memmap
            # If no units are provided we just assume standard units.
            self.target_calculator.\
                read_from_cube(snapshot["output"][0],
                               **target_calculator_kwargs)
            tmp_output = self.target_calculator.get_target()

        elif description["output"] is None:
            # In this case, only the input is processed.
            pass

        else:
            raise Exception(
                "Unknown file extension, cannot convert target"
                "data.")
        if description["output"] is not None:
            if get_rank() == 0:
                if output_path is not None:
                    np.save(output_path, tmp_output)

        if return_data:
            if description["input"] is not None and description["output"] \
                    is not None:
                return tmp_input, tmp_output
            elif description["input"] is None:
                return tmp_output
            elif description["output"] is None:
                return tmp_input

    def convert_snapshots(self, save_path="./",
                          naming_scheme="ELEM_snapshot*", starts_at=0,
                          file_based_communication=False,
                          descriptor_calculation_kwargs=None,
                          target_calculator_kwargs=None):
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

        target_calculator_kwargs : dict
            Dictionary with additional keyword arguments for the calculation
            or parsing of the target quantities.

        descriptor_calculation_kwargs : dict
            Dictionary with additional keyword arguments for the calculation
            or parsing of the descriptor quantities.
        """
        if descriptor_calculation_kwargs is None:
            descriptor_calculation_kwargs = {}

        if target_calculator_kwargs is None:
            target_calculator_kwargs = {}

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

            self.convert_single_snapshot(i,
                                         descriptor_calculation_kwargs,
                                         target_calculator_kwargs,
                                         input_path=os.path.join(save_path,
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
