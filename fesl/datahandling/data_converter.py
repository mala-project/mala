"""DataConverter class for converting snapshots into numpy arrays."""
from fesl.common.printout import printout
from fesl.common.parameters import ParametersData
from fesl.descriptors.descriptor_interface import DescriptorInterface
from fesl.targets.target_interface import TargetInterface
import numpy as np


class DataConverter:
    """
    Converts raw snapshots (direct DFT outputs) into numpy arrays for FESL.

    These snapshots can be e.g. Quantum Espresso results.

    Attributes
    ----------
    descriptor_calculator : fesl.descriptors.descriptor_base.DescriptorBase
        Descriptor calculator used for parsing/converting fingerprint data.

    target_calculator : fesl.targets.target_base.TargetBase
        Target calculator used for parsing/converting target data.
    """

    def __init__(self, parameters, descriptor_calculator=None,
                 target_calculator=None):
        """
        Create an instances of the DataConverter class.

        Parameters
        ----------
        parameters : fesl.common.parameters.Parameters
            The parameters object used for creating this instance.

        descriptor_calculator : fesl.descriptors.descriptor_base.DescriptorBase
            The descriptor calculator used for parsing/converting fingerprint
            data. If None, the descriptor calculator will be created by this
            object using the parameters provided. Default: None

        target_calculator : fesl.targets.target_base.TargetBase
            Target calculator used for parsing/converting target data. If None,
            the target calculator will be created by this object using the
            parameters provided.
        """
        self.parameters : ParametersData = parameters.data
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = TargetInterface(parameters)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = DescriptorInterface(parameters)

        self.__snapshots_to_convert = []
        self.__snapshot_description = []
        self.__snapshot_units = []

    def add_snapshot_qeout_cube(self, qe_out_file, qe_out_directory,
                                cube_naming_scheme, cube_directory,
                                input_units = None, output_units = None):
        """
        Add a Quantum Espresso snapshot to the list of conversion list.

        Please not that a Quantum Espresso snapshot consists of:

            - a Quantum Espresso output file for a self-consistent calculation.
            - multiple .cube files containing the LDOS

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        qe_out_directory : string
            Path to Quantum Espresso output file for snapshot.

        cube_naming_scheme : string
            Naming scheme for the LDOS .cube files.

        cube_directory : string
            Directory containing the LDOS .cube files.
        """
        self.__snapshots_to_convert.append([qe_out_file, qe_out_directory,
                                            cube_naming_scheme,
                                            cube_directory])
        self.__snapshot_description.append(["qe.out", ".cube"])
        self.__snapshot_units.append([input_units, output_units])

    def convert_single_snapshot(self, snapshot_number):
        """
        Convert single snapshot from the conversion lists.

        Returns the preprocessed data as numpy array, which might be
        beneficial during testing.

        Parameters
        ----------
        snapshot_number : integer
            Position of the desired snapshot in the snapshot list.

        Returns
        -------
        inputs : numpy.array
            Numpy array containing the preprocessed inputs.

        outputs : numpy.array
            Numpy array containing the preprocessed outputs.

        """
        snapshot = self.__snapshots_to_convert[snapshot_number]
        description = self.__snapshot_description[snapshot_number]
        original_units = self.__snapshot_units[snapshot_number]

        # Parse and/or calculate the input descriptors.
        if description[0] == "qe.out":
            if original_units[0] is None:
                tmp_input = self.descriptor_calculator. \
                    calculate_from_qe_out(snapshot[0], snapshot[1])
            else:
                tmp_input = self.descriptor_calculator. \
                    calculate_from_qe_out(snapshot[0], snapshot[1],
                                          units=original_units[0])

            # Cut the xyz information if requested by the user.
            if self.parameters.descriptors_contain_xyz is False:
                tmp_input = tmp_input[:, :, :, 3:]

        else:
            raise Exception("Unknown file extension, cannot convert target")

        # Parse and/or calculate the output descriptors.
        if description[1] == ".cube":

            # If no units are provided we just assume standard units.
            if original_units[1] is None:
                tmp_output = self.target_calculator.read_from_cube(snapshot[2],
                                                                   snapshot[3])
            else:
                tmp_output = self.target_calculator.\
                    read_from_cube(snapshot[2], snapshot[3], units=
                                   original_units[1])
        else:
            raise Exception("Unknown file extension, cannot convert target"
                            "data.")

        return tmp_input, tmp_output

    def convert_snapshots(self, save_path="./",
                          naming_scheme="ELEM_snapshot*"):
        """
        Convert the snapshots in the list to numpy arrays.

        These can then be used by FESL.

        Parameters
        ----------
        save_path : string
            Directory in which the snapshots will be saved.

        naming_scheme : string
            String detailing the naming scheme for the snapshots. * symbols
            will be replaced with the snapshot number.
        """
        for i in range(0, len(self.__snapshots_to_convert)):
            input_data, output_data = self.convert_single_snapshot(i)
            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(i))
            printout("Saving snapshot", i, "at ", save_path)
            np.save(save_path+snapshot_name+".in.npy", input_data)
            np.save(save_path+snapshot_name+".out.npy", output_data)
