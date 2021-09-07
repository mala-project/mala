"""SNAP descriptor class."""
import os

import warnings
import ase
import ase.io
from .lammps_utils import *
from .descriptor_base import DescriptorBase
try:
    from lammps import lammps
except ModuleNotFoundError:
    warnings.warn("You either don't have LAMMPS installed or it is not "
                  "configured correctly. Using SNAP descriptors "
                  "might still work, but trying to calculate SNAP "
                  "descriptors from atomic positions will crash.",
                  stacklevel=3)


class SNAP(DescriptorBase):
    """Class for calculation and parsing of SNAP descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(SNAP, self).__init__(parameters)
        self.in_format_ase = ""

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a SNAP descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.
        """
        if in_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for SNAP.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a SNAP descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for SNAP.")

    def calculate_from_qe_out(self, qe_out_file, qe_out_directory):
        """
        Calculate the SNAP descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        qe_out_directory : string
            Path to Quantum Espresso output file for snapshot.

        Returns
        -------
        snap_descriptors : numpy.array
            Numpy array containing the SNAP descriptors with the dimension
            (x,y,z,snap_dimension)

        """
        self.in_format_ase = "espresso-out"
        print("Calculating SNAP descriptors from", qe_out_file, "at",
              qe_out_directory)
        return self.__calculate_snap(os.path.join(qe_out_directory,
                                                  qe_out_file),
                                     qe_out_directory)

    def __calculate_snap(self, infile, outdir):
        """Perform actual SNAP calculation."""
        from lammps import lammps
        lammps_format = "lammps-data"
        # We get the atomic information by using ASE.
        atoms = ase.io.read(infile, format=self.in_format_ase)

        # Enforcing / Checking PBC on the read atoms.
        atoms = self.enforce_pbc(atoms)
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        # We also need to know how big the grid is.
        # Iterating directly through the file is slow, but the
        # grid information is at the top (around line 200).
        nx = None
        ny = None
        nz = None
        if len(self.dbg_grid_dimensions) == 3:
            nx = self.dbg_grid_dimensions[0]
            ny = self.dbg_grid_dimensions[1]
            nz = self.dbg_grid_dimensions[2]
        else:
            qe_outfile = open(infile, "r")
            lines = qe_outfile.readlines()
            for line in lines:
                if "FFT dimensions" in line:
                    tmp = line.split("(")[1].split(")")[0]
                    nx = int(tmp.split(",")[0])
                    ny = int(tmp.split(",")[1])
                    nz = int(tmp.split(",")[2])
                    break
        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = ["-screen", "none", "-log", os.path.join(outdir,
                                                               "lammps_log.tmp")]
        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs,
                                      {
                                        "ngridx": nx,
                                        "ngridy": ny,
                                        "ngridz": nz,
                                        "twojmax": self.parameters.twojmax,
                                        "rcutfac": self.parameters.rcutfac,
                                        "atom_config_fname": ase_out_path
                                      })

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # An empty string means that the user wants to use the standard input.
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("snap")[0]
            self.parameters.lammps_compute_file = os.path.join(filepath,
                                                               "in.bgrid.python")

        # Do the LAMMPS calculation.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (self.parameters.twojmax+2) * \
                 (self.parameters.twojmax+3)*(self.parameters.twojmax+4)
        ncoeff = ncoeff // 24   # integer division
        self.fingerprint_length = ncols0+ncoeff

        # Extract data from LAMMPS calculation.
        snap_descriptors_np = \
            extract_compute_np(lmp, "bgrid", 0, 2,
                               (nz, ny, nx, self.fingerprint_length))

        # switch from x-fastest to z-fastest order (swaps 0th and 2nd
        # dimension)
        snap_descriptors_np = snap_descriptors_np.transpose([2, 1, 0, 3])

        # The next two commands can be used to check whether the calculated
        # SNAP descriptors
        # are identical to the ones calculated with the Sandia workflow.
        # They generally are.
        # print("snap_descriptors_np shape = ",snap_descriptors_np.shape,
        # flush=True)
        # np.save(set[4]+"test.npy", snap_descriptors_np, allow_pickle=True)
        return snap_descriptors_np
