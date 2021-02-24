"""DataConverter class for converting snapshots into numpy arrays."""
from fesl.common.parameters import printout
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
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = TargetInterface(parameters)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = DescriptorInterface(parameters)

        self.__snapshots_to_convert = []

    def add_snapshot_qeout_cube(self, qe_out_file, qe_out_directory,
                                cube_naming_scheme, cube_directory):
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
        i = 0
        for snapshot in self.__snapshots_to_convert:
            ##############
            # Input data.
            # Calculate the SNAP descriptors from the QE calculation.
            ##############

            # print("Calculating descriptors for snapshot ", snapshot[0],
            # "at ", snapshot[1])
            # print(np.shape(self.descriptor_calculator.calculate_
            # from_qe_out(snapshot[0], snapshot[1])))

            ##############
            # Output data.
            # Read the LDOS data.
            ##############

            printout("Reading targets from ", snapshot[2], "at ", snapshot[3])
            tmp_input = self.target_calculator.read_from_cube(snapshot[2],
                                                              snapshot[3])

            # Here, raw_input only contains the file name given by ASE and the
            # dimensions of the grd.
            # These are the input parameters we need for LAMMPS.
            printout("Calculating descriptors for ", snapshot[0], "at ",
                     snapshot[1])
            tmp_output = self.descriptor_calculator.\
                calculate_from_qe_out(snapshot[1]+snapshot[0], snapshot[1])

            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(i))
            printout("Saving ", i, "at ", save_path)
            np.save(snapshot_name+".in.npy", tmp_input)
            np.save(snapshot_name+".out.npy", tmp_output)
            i += 1
