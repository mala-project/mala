"""DataConverter class for converting snapshots into numpy arrays."""
import os

import json

from mala.common.parallelizer import printout, get_rank, get_comm
from mala.common.parameters import ParametersData
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target
from mala.version import __version__ as mala_version

descriptor_input_types = [
    "espresso-out"
]
target_input_types = [
    ".cube", ".xsf"
]
additional_info_input_types = [
    "espresso-out"
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
        self.parameters_full = parameters
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
                     descriptor_units=None,
                     metadata_input_type=None,
                     metadata_input_path=None,
                     target_units=None):
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

        metadata_input_type : string
            Type of additional metadata to be processed.
            See mala.datahandling.data_converter.additional_info_input_types
            for options.
            This is essentially the same as additional_info_input_type,
            but will not affect saving; i.e., the data given here will
            only be saved in OpenPMD files, not saved separately.
            If additional_info_input_type is set, this argument will be
            ignored.

        metadata_input_path : string
            Path of additional metadata to be processed.
            See metadata_input_type for extended info on use.

        descriptor_units : string
            Units for descriptor data processing.

        target_units : string
            Units for target data processing.
        """
        # Check the input.
        if descriptor_input_type is not None:
            if descriptor_input_path is None:
                raise Exception(
                    "Cannot process descriptor data with no path "
                    "given.")
            if descriptor_input_type not in descriptor_input_types:
                raise Exception(
                    "Cannot process this type of descriptor data.")
            self.process_descriptors = True

        if target_input_type is not None:
            if target_input_path is None:
                raise Exception("Cannot process target data with no path "
                                "given.")
            if target_input_type not in target_input_types:
                raise Exception("Cannot process this type of target data.")
            self.process_targets = True

        if additional_info_input_type is not None:
            metadata_input_type = additional_info_input_type
            if additional_info_input_path is None:
                raise Exception("Cannot process additional info data with "
                                "no path given.")
            if additional_info_input_type not in additional_info_input_types:
                raise Exception(
                    "Cannot process this type of additional info "
                    "data.")
            self.process_additional_info = True

            metadata_input_path = additional_info_input_path

        if metadata_input_type is not None:
            if metadata_input_path is None:
                raise Exception("Cannot process additional info data with "
                                "no path given.")
            if metadata_input_type not in additional_info_input_types:
                raise Exception(
                    "Cannot process this type of additional info "
                    "data.")

        # Assign info.
        self.__snapshots_to_convert.append({"input": descriptor_input_path,
                                            "output": target_input_path,
                                            "additional_info":
                                                additional_info_input_path,
                                            "metadata": metadata_input_path})
        self.__snapshot_description.append({"input": descriptor_input_type,
                                            "output": target_input_type,
                                            "additional_info":
                                                additional_info_input_type,
                                            "metadata": metadata_input_type})
        self.__snapshot_units.append({"input": descriptor_units,
                                      "output": target_units})

    def convert_snapshots(self, complete_save_path=None,
                          descriptor_save_path=None,
                          target_save_path=None,
                          additional_info_save_path=None,
                          naming_scheme="ELEM_snapshot*.npy", starts_at=0,
                          file_based_communication=False,
                          descriptor_calculation_kwargs=None,
                          target_calculator_kwargs=None,
                          use_fp64=False):
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

        use_fp64 : bool
            If True, data is saved with double precision. If False (default),
            single precision (FP32) is used. This is advantageous, since
            internally, torch models are constructed with FP32 anyway.
        """
        if "." in naming_scheme:
            file_ending = naming_scheme.split(".")[-1]
            naming_scheme = naming_scheme.split(".")[0]
            if file_ending != "npy":
                import openpmd_api as io

                if file_ending not in io.file_extensions:
                    raise Exception("Invalid file ending selected: " +
                                    file_ending)
        else:
            file_ending = "npy"

        if file_ending == "npy":
            # I will leave the deprecation warning out for now, we re-enable
            # it as soon as we have a precise timeline.
            # parallel_warn("NumPy array based file saving will be deprecated"
            #               "starting in MALA v1.3.0.", min_verbosity=0,
            #               category=FutureWarning)
            pass
        else:
            file_based_communication = False

        if descriptor_calculation_kwargs is None:
            descriptor_calculation_kwargs = {}

        if target_calculator_kwargs is None:
            target_calculator_kwargs = {}

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

        if file_ending != "npy":
            snapshot_name = naming_scheme
            series_name = snapshot_name.replace("*", str("%01T"))

            if self.process_descriptors:
                if self.parameters._configuration["mpi"]:
                    input_series = io.Series(
                        os.path.join(descriptor_save_path,
                                     series_name + ".in." + file_ending),
                        io.Access.create,
                        get_comm(),
                        options=json.dumps(
                            self.parameters_full.openpmd_configuration))
                else:
                    input_series = io.Series(
                        os.path.join(descriptor_save_path,
                                     series_name + ".in." + file_ending),
                        io.Access.create,
                        options=json.dumps(
                            self.parameters_full.openpmd_configuration))
                input_series.set_attribute("is_mala_data", 1)
                input_series.set_software(name="MALA", version="x.x.x")
                input_series.author = "..."

            if self.process_targets:
                if self.parameters._configuration["mpi"]:
                    output_series = io.Series(
                        os.path.join(target_save_path,
                                     series_name + ".out." + file_ending),
                        io.Access.create,
                        get_comm(),
                        options=json.dumps(
                            self.parameters_full.openpmd_configuration))
                else:
                    output_series = io.Series(
                        os.path.join(target_save_path,
                                     series_name + ".out." + file_ending),
                        io.Access.create,
                        options=json.dumps(
                            self.parameters_full.openpmd_configuration))

                output_series.set_attribute("is_mala_data", 1)
                output_series.set_software(name="MALA", version=mala_version)
                output_series.author = "..."

        for i in range(0, len(self.__snapshots_to_convert)):
            snapshot_number = i + starts_at
            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(snapshot_number))

            # Create the paths as needed.
            if self.process_additional_info:
                info_path = os.path.join(additional_info_save_path,
                                         snapshot_name + ".info.json")
            else:
                info_path = None
            input_iteration = None
            output_iteration = None

            if file_ending == "npy":
                # Create the actual paths, if needed.
                if self.process_descriptors:
                    descriptor_path = os.path.join(descriptor_save_path,
                                                   snapshot_name + ".in." +
                                                   file_ending)
                else:
                    descriptor_path = None

                memmap = None
                if self.process_targets:
                    target_path = os.path.join(target_save_path,
                                               snapshot_name + ".out."+
                                               file_ending)
                    # A memory mapped file is used as buffer for distributed cases.
                    if self.parameters._configuration["mpi"] and \
                            file_based_communication:
                        memmap = os.path.join(target_save_path, snapshot_name +
                                              ".out.npy_temp")
                else:
                    target_path = None
            else:
                descriptor_path = None
                target_path = None
                memmap = None
                if self.process_descriptors:
                    input_iteration = input_series.write_iterations()[i + starts_at]
                    input_iteration.dt = i + starts_at
                    input_iteration.time = 0
                if self.process_targets:
                    output_iteration = output_series.write_iterations()[i + starts_at]
                    output_iteration.dt = i + starts_at
                    output_iteration.time = 0

            self.__convert_single_snapshot(i, descriptor_calculation_kwargs,
                                           target_calculator_kwargs,
                                           input_path=descriptor_path,
                                           output_path=target_path,
                                           use_memmap=memmap,
                                           input_iteration=input_iteration,
                                           output_iteration=output_iteration,
                                           additional_info_path=info_path,
                                           use_fp64=use_fp64)

            if get_rank() == 0:
                if self.parameters._configuration["mpi"] \
                        and file_based_communication:
                    os.remove(memmap)

        # Properly close series
        if file_ending != "npy":
            if self.process_descriptors:
                del input_series
            if self.process_targets:
                del output_series

    def __convert_single_snapshot(self, snapshot_number,
                                  descriptor_calculation_kwargs,
                                  target_calculator_kwargs,
                                  input_path=None,
                                  output_path=None,
                                  additional_info_path=None,
                                  use_memmap=None,
                                  output_iteration=None,
                                  input_iteration=None,
                                  use_fp64=False):
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

        output_iteration : OpenPMD iteration
            OpenPMD iteration to be used to save the output data of the current
            snapshot, as part of an OpenPMD Series.

        input_iteration : OpenPMD iteration
            OpenPMD iteration to be used to save the input data of the current
            snapshot, as part of an OpenPMD Series.

        use_fp64 : bool
            If True, data is saved with double precision. If False (default),
            single precision (FP32) is used. This is advantageous, since
            internally, torch models are constructed with FP32 anyway.

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
        if description["input"] == "espresso-out":
            descriptor_calculation_kwargs["units"] = original_units["input"]
            descriptor_calculation_kwargs["use_fp64"] = use_fp64

            tmp_input, local_size = self.descriptor_calculator. \
                calculate_from_qe_out(snapshot["input"],
                                      **descriptor_calculation_kwargs)

        elif description["input"] is None:
            # In this case, only the output is processed.
            pass

        else:
            raise Exception("Unknown file extension, cannot convert descriptor")

        if description["input"] is not None:
            # Save data and delete, if not requested otherwise.
            if input_path is not None and input_iteration is None:
                if self.parameters._configuration["mpi"]:
                    tmp_input = self.descriptor_calculator. \
                        gather_descriptors(tmp_input)
                if get_rank() == 0:
                    self.descriptor_calculator.\
                        write_to_numpy_file(input_path, tmp_input)
            else:
                if self.parameters._configuration["mpi"]:
                    tmp_input, local_offset, local_reach = \
                        self.descriptor_calculator.convert_local_to_3d(tmp_input)
                    self.descriptor_calculator. \
                        write_to_openpmd_iteration(input_iteration,
                                                   tmp_input,
                                                   local_offset=local_offset,
                                                   local_reach=local_reach)
                else:
                    self.descriptor_calculator. \
                        write_to_openpmd_iteration(input_iteration,
                                                   tmp_input)
            del tmp_input

        ###########
        # Outputs #
        ###########

        if description["output"] is not None:
            if output_path is not None and output_iteration is None:
                # Parse and/or calculate the output descriptors.
                if description["output"] == ".cube":
                    target_calculator_kwargs["units"] = original_units[
                        "output"]
                    target_calculator_kwargs["use_memmap"] = use_memmap
                    target_calculator_kwargs["use_fp64"] = use_fp64

                    # If no units are provided we just assume standard units.
                    tmp_output = self.target_calculator. \
                        read_from_cube(snapshot["output"],
                                       **target_calculator_kwargs)

                elif description["output"] == ".xsf":
                    target_calculator_kwargs["units"] = original_units[
                        "output"]
                    target_calculator_kwargs["use_memmap"] = use_memmap
                    target_calculator_kwargs["use_fp664"] = use_fp64

                    # If no units are provided we just assume standard units.
                    tmp_output = self.target_calculator. \
                        read_from_xsf(snapshot["output"],
                                      **target_calculator_kwargs)

                elif description["output"] is None:
                    # In this case, only the input is processed.
                    pass

                else:
                    raise Exception(
                        "Unknown file extension, cannot convert target"
                        "data.")

                if get_rank() == 0:
                    self.target_calculator.write_to_numpy_file(output_path,
                                                               tmp_output)
            else:
                metadata = None
                if description["metadata"] is not None:
                    metadata = [snapshot["metadata"],
                                description["metadata"]]
                # Parse and/or calculate the output descriptors.
                if self.parameters._configuration["mpi"]:
                    target_calculator_kwargs["return_local"] = True
                if description["output"] == ".cube":
                    target_calculator_kwargs["units"] = original_units[
                        "output"]
                    target_calculator_kwargs["use_memmap"] = use_memmap
                    # If no units are provided we just assume standard units.
                    tmp_output = self.target_calculator. \
                        read_from_cube(snapshot["output"],
                                       **target_calculator_kwargs)

                elif description["output"] == ".xsf":
                    target_calculator_kwargs["units"] = original_units[
                        "output"]
                    target_calculator_kwargs["use_memmap"] = use_memmap
                    # If no units are provided we just assume standard units.
                    tmp_output = self.target_calculator. \
                        read_from_xsf(snapshot["output"],
                                      **target_calculator_kwargs)

                elif description["output"] is None:
                    # In this case, only the input is processed.
                    pass

                else:
                    raise Exception(
                        "Unknown file extension, cannot convert target"
                        "data.")

                if self.parameters._configuration["mpi"]:
                    self.target_calculator. \
                        write_to_openpmd_iteration(output_iteration,
                                                   tmp_output[0],
                                                   feature_from=tmp_output[1],
                                                   feature_to=tmp_output[2],
                                                   additional_metadata=metadata)
                else:
                    self.target_calculator. \
                        write_to_openpmd_iteration(output_iteration,
                                                   tmp_output,
                                                   additional_metadata=metadata)
            del tmp_output

        # Parse and/or calculate the additional info.
        if description["additional_info"] is not None:
            # Parsing and saving is done using the target calculator.
            self.target_calculator. \
                read_additional_calculation_data(snapshot["additional_info"],
                                                 description["additional_info"])
            self.target_calculator. \
                write_additional_calculation_data(additional_info_path)
