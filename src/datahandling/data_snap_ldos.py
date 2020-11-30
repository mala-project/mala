'''
Collects / calculates SNAP descriptors and LDOS data from a DFT calculation.
Uses LAMMPS, so make sure it is available.
'''
import ase
import ase.io


class data_snap_ldos(data_base):
    """Mock-Up data class that is supposed to work like the actual data classes going to be used for FP/LDOS data,
    but operates on the MNIST data."""
    def __init__(self, p):
        self.parameters = p.data
        self.raw_input = []
        self.raw_output = []
        self.training_data_raw = []
        self.validation_data_raw = []
        self.test_data_raw = []

        self.training_data_inputs = []
        self.validation_data_inputs = []
        self.test_data_inputs = []

        self.training_data_targets = []
        self.validation_data_targets = []
        self.test_data_targets = []

        self.training_data_set = []
        self.validation_data_set = []
        self.test_data_set = []

        self.nr_training_data = 0
        self.nr_test_data = 0
        self.nr_validation_data = 0

    def load_data(self):
        """Loads FP/SNAP/LDOS data. Data can be loaded as FP (i.e. atomic positions including information about the grid)
        or already calculated SNAP descriptors."""

        # Determine on which kind of raw data we are operating on.
        if (self.parameters.datatype == "QE+LDOS"):
            for sub in self.parameters.qe_calc_list
            self.raw_input = get_SNAP_from_QE()
        else:
            raise Exception("Unsupported type of data. This data handler can only operate with QuantumEspresso and LDOS data at the moment.")

    def get_SNAP_from_QE(self):
        """Parses a QE outfile into a format that can be used by LAMMPS,
        then uses LAMMPS to calculate SNAP descriptors."""
        qe_format = "espresso-out"

        # iteration over all subdirectories / files provided by the user.


        atoms = ase.io.read(qe_filepath, format=qe_format)
        ase.io.write()




if __name__ == "__main__":
    raise Exception(
        "data_snap_ldos.py - test of basic functions not yet implemented.")
