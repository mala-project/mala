from .handler_base import HandlerBase
import numpy as np
from fesl.common.parameters import printout


class HandlerQEoutCube(HandlerBase):
    """Collects data from a QuantumEspresso calculation.
    Input data is read by reading the QuantumEspresso outfile.
    Output data is read by parsing multiple *.cube files.
    """

    def __init__(self, p, input_data_scaler, output_data_scaler, descriptor_calculator, target_parser):
        super(HandlerQEoutCube, self).__init__(p, input_data_scaler, output_data_scaler, descriptor_calculator,
                                               target_parser)

    def add_snapshot(self, qe_out_file, qe_out_directory, cube_naming_scheme, cube_directory,
                     input_units=None, output_units="1/eV"):
        """Adds a snapshot to data handler. For this type of data,
        a QuantumEspresso outfile, an outfile from the LDOS calculation and
        a directory containing the cube files"""
        # FIXME: Use the new snapshot class instead.
        self.parameters.snapshot_directories_list.append(
            [qe_out_file, qe_out_directory, cube_naming_scheme, cube_directory])

    def load_data(self):
        """Loads data and transforms it into descriptor / target data on the grid.
        At the end of this function we have two pytorch tensors holding the input and the output data."""

        # Load from every snapshot directory.
        for snapshot in self.parameters.snapshot_directories_list:
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

            printout("Reading targets for snapshot ", snapshot[2], "at ", snapshot[3])
            self.target_parser.read_from_cube(snapshot[2], snapshot[3])

            # Here, raw_input only contains the file name given by ASE and the dimensions of the grd.
            # These are the input parameters we need for LAMMPS.
            # self.raw_input.append(self.get_atoms_from_qe(self.parameters.directory+sub))

    # def prepare_data(self):
    #     """Cuts input/output data into training, validation and test set.
    #     then adds the training, validation and test data to pytorch datahandlers
    #     to be used by the network."""

    def get_input_dimension(self):
        if self.descriptor_calculator.fingerprint_length > 0:
            return self.descriptor_calculator.fingerprint_length
        else:
            raise Exception("No descriptors were calculated, cannot give input dimension.")

    def get_output_dimension(self):
        if self.parameters.ldos_gridsize > 0:
            return self.target_parser.target_length
        else:
            raise Exception("No targets were read, cannot give output dimension.")
