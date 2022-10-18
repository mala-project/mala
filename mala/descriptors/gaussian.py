"""Gaussian descriptor class."""
import os

import ase
import ase.io
try:
    from lammps import lammps
    # For version compatibility; older lammps versions (the serial version
    # we still use on some machines) do not have these constants.
    try:
        from lammps import constants as lammps_constants
    except ImportError:
        pass
except ModuleNotFoundError:
    pass

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.common.parallelizer import printout


class GaussianDescriptors(Descriptor):
    """Class for calculation and parsing of Gaussian descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(GaussianDescriptors, self).__init__(parameters)
        self.in_format_ase = ""
        self.grid_dimensions = []
        self.verbosity = parameters.verbosity

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a Gaussian descriptor.

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
            raise Exception("Unsupported unit for Gaussian descriptors.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a Gaussian descriptor.

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
            raise Exception("Unsupported unit for Gaussian descriptors.")

    def calculate_from_qe_out(self, qe_out_file, qe_out_directory,
                              working_directory=None,
                              **kwargs):
        """
        Calculate the Gaussian descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        qe_out_directory : string
            Path to Quantum Espresso output file for snapshot.

        working_directory : string
            A directory in which to perform the LAMMPS calculation.
            Optional, if None, the QE out directory will be used.

        Returns
        -------
        snap_descriptors : numpy.array
            Numpy array containing the Gaussian descriptors with the dimension
            (x,y,z,snap_dimension)

        """
        self.in_format_ase = "espresso-out"
        printout("Calculating Gaussian descriptors from", qe_out_file, "at",
                 qe_out_directory, min_verbosity=0)
        # We get the atomic information by using ASE.
        infile = os.path.join(qe_out_directory, qe_out_file)
        atoms = ase.io.read(infile, format=self.in_format_ase)

        # Enforcing / Checking PBC on the read atoms.
        atoms = self.enforce_pbc(atoms)

        # Get the grid dimensions.
        qe_outfile = open(infile, "r")
        lines = qe_outfile.readlines()
        nx = 0
        ny = 0
        nz = 0

        for line in lines:
            if "FFT dimensions" in line:
                tmp = line.split("(")[1].split(")")[0]
                nx = int(tmp.split(",")[0])
                ny = int(tmp.split(",")[1])
                nz = int(tmp.split(",")[2])
                break

        if working_directory is None:
            working_directory = qe_out_directory

        return self.__calculate_gaussian_descriptors(atoms,
                                                     working_directory,
                                                     [nx, ny, nz])

    def calculate_from_atoms(self, atoms, grid_dimensions,
                             working_directory="."):
        """
        Calculate the Gaussian descriptors based on the atomic configurations.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object holding the atomic configuration.

        grid_dimensions : list
            Grid dimensions to be used, in the format [x,y,z].

        working_directory : string
            A directory in which to perform the LAMMPS calculation.

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)
        """
        # Enforcing / Checking PBC on the input atoms.
        atoms = self.enforce_pbc(atoms)
        return self.__calculate_gaussian_descriptors(atoms, working_directory,
                                                     grid_dimensions)

    def __calculate_gaussian_descriptors(self, atoms, outdir, grid_dimensions):
        """Perform actual Gaussian descriptor calculation."""
        from lammps import lammps
        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        if len(self.dbg_grid_dimensions) == 3:
            nx = self.dbg_grid_dimensions[0]
            ny = self.dbg_grid_dimensions[1]
            nz = self.dbg_grid_dimensions[2]
        else:
            nx = grid_dimensions[0]
            ny = grid_dimensions[1]
            nz = grid_dimensions[2]

        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = ["-screen", "none", "-log",
                       os.path.join(outdir, "lammps_ggrid_log.tmp")]

        # LAMMPS processor grid filled by parent class.
        lammps_dict = self._setup_lammps_processors(nx, ny, nz)

        # Set the values not already filled in the LAMMPS setup.
        lammps_dict["sigma"] = self.parameters.gaussian_descriptors_sigma
        lammps_dict["rcutfac"] = self.parameters.gaussian_descriptors_cutoff
        lammps_dict["atom_config_fname"] = ase_out_path

        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs, lammps_dict)

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # For now the file is chosen automatically, because this is used
        # mostly under the hood anyway.
        filepath = __file__.split("gaussian")[0]
        if self.parameters._configuration["mpi"]:
            if self.parameters.use_z_splitting:
                runfile = os.path.join(filepath, "in.ggrid.python")
            else:
                runfile = os.path.join(filepath, "in.ggrid_defaultproc.python")
        else:
            runfile = os.path.join(filepath, "in.ggrid_defaultproc.python")
        lmp.file(runfile)

        # Extract the data.
        nrows_ggrid = extract_compute_np(lmp, "ggrid",
                                         lammps_constants.LMP_STYLE_LOCAL,
                                         lammps_constants.LMP_SIZE_ROWS)
        ncols_ggrid = extract_compute_np(lmp, "ggrid",
                                         lammps_constants.LMP_STYLE_LOCAL,
                                         lammps_constants.LMP_SIZE_COLS)

        gaussian_descriptors_np = \
            extract_compute_np(lmp, "ggrid",
                               lammps_constants.LMP_STYLE_LOCAL, 2,
                               array_shape=(nrows_ggrid, ncols_ggrid))

        if self.descriptors_contain_xyz:
            return gaussian_descriptors_np.copy(), nrows_ggrid
        else:
            return gaussian_descriptors_np[:, 6:].copy(), nrows_ggrid
