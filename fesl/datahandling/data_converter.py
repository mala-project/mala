from fesl.common.parameters import printout
from fesl.descriptors.descriptor_interface import DescriptorInterface
from fesl.targets.target_interface import TargetInterface
import numpy as np

class DataConverter:
    """
    This class can use a couple of raw snapshots (i.e. direct outputs of QE) and transform them into snapshots
    the FESL DataHandler class can use easily.
    """

    def __init__(self, p, descriptor_calculator=None, target_calculator=None):
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = TargetInterface(p)

        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = DescriptorInterface(p)

        self.snapshots_to_convert = []

    def add_snapshot_qeout_cube(self, qe_out_file, qe_out_directory, cube_naming_scheme, cube_directory):
        self.snapshots_to_convert.append([qe_out_file, qe_out_directory, cube_naming_scheme, cube_directory])

    def convert_snapshots(self, save_path="./", naming_scheme="ELEM_snapshot*"):
        i = 0
        for snapshot in self.snapshots_to_convert:
            ##############
            # Input data.
            # Calculate the SNAP descriptors from the QE calculation.
            ##############

            # print("Calculating descriptors for snapshot ", snapshot[0], "at ", snapshot[1])
            # print(np.shape(self.descriptor_calculator.calculate_from_qe_out(snapshot[0], snapshot[1])))

            ##############
            # Output data.
            # Read the LDOS data.
            ##############

            printout("Reading targets from ", snapshot[2], "at ", snapshot[3])
            tmp_input = self.target_calculator.read_from_cube(snapshot[2], snapshot[3])


            # Here, raw_input only contains the file name given by ASE and the dimensions of the grd.
            # These are the input parameters we need for LAMMPS.
            printout("Calculating descriptors for ", snapshot[0], "at ", snapshot[1])
            tmp_output = self.descriptor_calculator.calculate_snap(snapshot[1]+snapshot[0], snapshot[1])

            snapshot_name = naming_scheme
            snapshot_name = snapshot_name.replace("*", str(i))
            printout("Saving ", i, "at ", save_path)
            np.save(snapshot_name+".in.npy", tmp_input)
            np.save(snapshot_name+".out.npy", tmp_output)
            i += 1