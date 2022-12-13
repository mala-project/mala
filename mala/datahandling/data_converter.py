"""DataConverter class for converting snapshots into numpy arrays."""
import os

import numpy as np

from mala.common.parallelizer import printout, get_rank
from mala.common.parameters import ParametersData
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target

descriptor_input_types = [
    "qe.out"
]
target_input_types = [
    ".cube", ".xsf"
]
additional_info_input_types = [
    "qe.out"
]


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

        # Keep track of what has to be done by this data converter.
        self.process_descriptors = False
        self.process_targets = False
        self.process_additional_info = False

    def add_snapshot(self, descriptor_input_type=None,
                     descriptor_input_path=None,
                     target_input_type=None,
                     target_input_path=None,
                     additional_info_input_type=None,
                     additional_info_input_path=None,
                     descriptor_units=None, target_units=None):
        """
        Add a snapshot to be processed.

        Parameters
        ----------
        descriptor_input_type : string
            Type of descriptor data to be processed.
            See mala.datahandling.data_converter.descriptor_input_types
            for options.

        descriptor_input_path : string
            Path of descriptor data to be processed.

        target_input_type : string
            Type of target data to be processed.
            See mala.datahandling.data_converter.target_input_types
            for options.

        target_input_path : string
            Path of target data to be processed.

        additional_info_input_type : string
            Type of additional info data to be processed.
            See mala.datahandling.data_converter.additional_info_input_types
            for options.

        additional_info_input_path : string
            Path of additional info data to be processed.

        descriptor_units : string
            Units for descriptor data processing.

        target_units : string
            Units for target data processing.
        """
        # Check the input.
        if descriptor_input_type is not None:
            if descriptor_input_path is None:
                raise Exception("Cannot process descriptor data with no path "
                                "given.")
            if descriptor_input_type not in descriptor_input_types:
                raise Exception("Cannot process this type of descriptor data.")
            self.process_descriptors = True

        if target_input_type is not None:
            if target_input_path is None:
                raise Exception("Cannot process target data with no path "
                                "given.")
            if target_input_type not in target_input_types:
                raise Exception("Cannot process this type of target data.")
            self.process_targets = True

        if additional_info_input_type is not None:
            if additional_info_input_path is None:
                raise Exception("Cannot process additional info data with "
                                "no path given.")
            if additional_info_input_type not in additional_info_input_types:
                raise Exception("Cannot process this type of additional info "
                                "data.")
            self.process_additional_info = True

        # Assign info.
        self.__snapshots_to_convert.append({"input": descriptor_input_path,
                                            "output": target_input_path,
                                            "additional_info":
                                                additional_info_input_path})
        self.__snapshot_description.append({"input": descriptor_input_type,
                                            "output": target_input_type,
                                            "additional_info":
                                                additional_info_input_type})
        self.__snapshot_units.append({"input": descriptor_units,
                                      "output": target_units})

    def convert_single_snapshot(self, snapshot_number,
                                descriptor_calculation_kwargs,
                                target_calculator_kwargs,
                                input_path=None,
                                output_path=None,
                                additional_info_path=None,
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
                calculate_from_qe_out(snapshot["input"],
                                      **descriptor_calculation_kwargs)
            if self.parameters._configuration["mpi"]:
                tmp_input = self.descriptor_calculator.\
                    gather_descriptors(tmp_input)

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
                read_from_cube(snapshot["output"],
                               **target_calculator_kwargs)
            tmp_output = self.target_calculator.get_target()

        if description["output"] == ".xsf":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = use_memmap
            # If no units are provided we just assume standard units.
            self.target_calculator.\
                read_from_xsf(snapshot["output"],
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

        # Parse and/or calculate the additional info.
        if description["additional_info"] == "qe.out":
            # Parsing and saving is done using the target calculator.
            self.target_calculator.\
                read_additional_calculation_data("qe.out",
                                                 snapshot["additional_info"])
            self.target_calculator.\
                write_additional_calculation_data(additional_info_path)

        elif description["additional_info"] is None:
            # Not additional info provided, pass.
            pass
        else:
            raise Exception(
                "Unknown file extension, cannot convert additional info "
                "data.")

        if return_data:
            if description["input"] is not None and description["output"] \
                    is not None:
                return tmp_input, tmp_output
            elif description["input"] is None:
                return tmp_output
            elif description["output"] is None:
                return tmp_input

    def convert_snapshots(self, complete_save_path=None,
                          descriptor_save_path=None,
                          target_save_path=None,
                          additional_info_save_path=None,
                          naming_scheme="ELEM_snapshot*", starts_at=0,
                          file_based_communication=False,
                          descriptor_calculation_kwargs=None,
                          target_calculator_kwargs=None):
        """
        Convert the snapshots in the list to numpy arrays.

        These can then be used by MALA.

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

        additional_info_save_path : string
            Directory in which to save additional info data.

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
        if complete_save_path is not None:
            descriptor_save_path = complete_save_path
            target_save_path = complete_save_path
            additional_info_save_path = complete_save_path
        else:
            if self.process_targets is True and target_save_path is None:
                raise Exception("No target path specified, cannot process "
                                "data.")
            if self.process_descriptors is True and descriptor_save_path is None:
                raise Exception("No descriptor path specified, cannot "
                                "process data.")
            if self.process_additional_info is True and additional_info_save_path is None:
                raise Exception("No additional info path specified, cannot "
                                "process data.")

        if descriptor_calculation_kwargs is None:
            descriptor_calculation_kwargs = {}

        if target_calculator_kwargs is None:
            target_calculator_kwargs = {}

        for i in range(0, len(self.__snapshots_to_convert)):
            snapshot_number = i + starts_at
            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(snapshot_number))

            # Create the actual paths, if needed.
            if self.process_descriptors:
                descriptor_path = os.path.join(descriptor_save_path,
                                         snapshot_name+".in.npy")
            else:
                descriptor_path = None

            memmap = None
            if self.process_targets:
                target_path = os.path.join(target_save_path,
                                         snapshot_name+".out.npy")
                # A memory mapped file is used as buffer for distributed cases.
                if self.parameters._configuration["mpi"] and \
                        file_based_communication:
                    memmap = os.path.join(target_save_path, snapshot_name +
                                          ".out.npy_temp")
            else:
                target_path = None

            if self.process_additional_info:
                info_path = os.path.join(additional_info_save_path,
                                         snapshot_name+".info.json")
            else:
                info_path = None


            self.convert_single_snapshot(i,
                                         descriptor_calculation_kwargs,
                                         target_calculator_kwargs,
                                         input_path=descriptor_path,
                                         output_path=target_path,
                                         additional_info_path=info_path,
                                         use_memmap=memmap)
            printout("Saved snapshot", snapshot_number, min_verbosity=0)

            if get_rank() == 0:
                if self.parameters._configuration["mpi"] \
                        and file_based_communication:
                    os.remove(memmap)
