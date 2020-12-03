'''
Collects / calculates SNAP descriptors and LDOS data from a DFT calculation.
Uses LAMMPS, so make sure it is available.
'''
import ase
import ase.io
from .data_base import data_base
import os
from pathlib import Path
from .lammps_utils import *
from lammps import lammps


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

        self.fingerprint_length = 0

    def load_data(self):
        """Loads FP/SNAP/LDOS data. Data can be loaded as FP (i.e. atomic positions including information about the grid)
        or already calculated SNAP descriptors."""

        # Determine on which kind of raw data we are operating on.
        if (self.parameters.datatype == "QE+LDOS"):
            for sub in self.parameters.qe_calc_list:
                print("Loading data from QE calculation at ", sub)
                # Here, raw_input only contains the file name given by ASE and the dimensions of the grd.
                # These are the input parameters we need for LAMMPS.
                self.raw_input.append(self.get_atoms_from_qe(self.parameters.directory+sub))
        else:
            raise Exception("Unsupported type of data. This data handler can only operate with QuantumEspresso and LDOS data at the moment.")

    def prepare_data(self):
        """Calculates SNAP from atom/grid information, if necessary."""

        # Determine on which kind of raw data we are operating on.
        if (self.parameters.datatype == "QE+LDOS"):
            for set in self.raw_input:
                print("Calculating SNAP from atom/grid information at", set[0])
                snap_data = self.get_snap_from_atoms(set)
                # Now we have the SNAP and the LDOS data and have to put them together to input
                # them into pytorch.
        else:
            raise Exception("Unsupported type of data. This data handler can only operate with QuantumEspresso and LDOS data at the moment.")

    def get_atoms_from_qe(self, dirpath):
        """Parses a QE outfile into a format that can be used by LAMMPS,
        then uses LAMMPS to calculate SNAP descriptors."""
        qe_format = "espresso-out"
        lammps_format = "lammps-data"

        # We always use the LAST MODIFIED *.out file.
        # Yes, this is prone to error and has to be modified later.
        files = sorted(Path(dirpath).glob("*.out"), key=os.path.getmtime)

        # We get the atomic information by using ASE.
        atoms = ase.io.read(files[-1], format=qe_format)
        ase_out_path = dirpath+"lammps_input.tmp"
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        # We also need to know how big the grid is.
        # Iterating directly through the file is slow, but the
        # grid information is at the top (around line 200).
        if (len(self.parameters.dbg_grid_dimensions)==3):
            nx = self.parameters.dbg_grid_dimensions[0]
            ny = self.parameters.dbg_grid_dimensions[1]
            nz = self.parameters.dbg_grid_dimensions[2]
        else:
            qe_outfile = open(files[-1], "r")
            lines = qe_outfile.readlines()
            for line in lines:
                if "FFT dimensions" in line:
                    tmp = line.split("(")[1].split(")")[0]
                    nx = int(tmp.split(",")[0])
                    ny = int(tmp.split(",")[1])
                    nz = int(tmp.split(",")[2])
                    break
        return [ase_out_path, nx, ny, nz, dirpath]

    def get_snap_from_atoms(self, set):
        """Uses an ASE atoms file and grid informatsion to calculate SNAP descriptors"""

        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = ["-screen", "none", "-log", set[4]+"lammps_log.tmp"]
        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs,
            {
            "ngridx":set[1],
            "ngridy":set[2],
            "ngridz":set[3],
            "twojmax":self.parameters.twojmax,
            "rcutfac":self.parameters.rcutfac,
            "atom_config_fname":set[0]
            }
        )

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # An empty string means that the user wants to use the standard input.
        if (self.parameters.lammps_compute_file == ""):
            filepath = __file__.split("data_snap_ldos")[0]
            self.parameters.lammps_compute_file = filepath+"in.bgrid.python"

        # Do the LAMMPS calculation.
        try:
            lmp.file(self.parameters.lammps_compute_file)
        except lammps.LAMMPSException:
            raise Exception("There was a problem during the SNAP calculation. Exiting.")


        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (self.parameters.twojmax+2)*(self.parameters.twojmax+3)*(self.parameters.twojmax+4)
        ncoeff = ncoeff // 24 # integer division
        self.fingerprint_length = ncols0+ncoeff

        # Extract data from LAMMPS calculation.
        snap_descriptors_np = extract_compute_np(lmp, "bgrid", 0, 2, (set[3],set[2],set[1],self.fingerprint_length))

        # switch from x-fastest to z-fastest order (swaps 0th and 2nd dimension)
        snap_descriptors_np = snap_descriptors_np.transpose([2,1,0,3])

        # The next two commands can be used to check whether the calculated SNAP descriptors
        # are identical to the ones calculated with the Sandia workflow. They generally are.
        # print("snap_descriptors_np shape = ",snap_descriptors_np.shape, flush=True)
        # np.save(set[4]+"test.npy", snap_descriptors_np, allow_pickle=True)

        return snap_descriptors_np

    def get_input_dimension(self):
        if (self.fingerprint_length != 0):
            return self.fingerprint_length
        else:
            raise Exception("No fingerprints were calculated, cannot give input dimension.")

if __name__ == "__main__":
    raise Exception(
        "data_snap_ldos.py - test of basic functions not yet implemented.")
