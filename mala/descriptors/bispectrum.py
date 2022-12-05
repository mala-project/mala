"""SNAP descriptor class."""
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
import numpy as np

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.common.parallelizer import get_comm, printout, get_rank, get_size, \
    barrier


class Bispectrum(Descriptor):
    """Class for calculation and parsing of SNAP descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(Bispectrum, self).__init__(parameters)

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

    def _calculate(self, atoms, outdir, grid_dimensions, **kwargs):
        """Perform actual SNAP calculation."""
        from lammps import lammps
        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        # We also need to know how big the grid is.
        # Iterating directly through the file is slow, but the
        # grid information is at the top (around line 200).
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
                       os.path.join(outdir, "lammps_bgrid_log.tmp")]

        # LAMMPS processor grid filled by parent class.
        lammps_dict = self._setup_lammps_processors(nx, ny, nz)

        # Set the values not already filled in the LAMMPS setup.
        lammps_dict["twojmax"] = self.parameters.bispectrum_twojmax
        lammps_dict["rcutfac"] = self.parameters.bispectrum_cutoff
        lammps_dict["atom_config_fname"] = ase_out_path

        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs, lammps_dict)

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("bispectrum")[0]
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    self.parameters.lammps_compute_file = \
                        os.path.join(filepath, "in.bgridlocal.python")
                else:
                    self.parameters.lammps_compute_file = \
                        os.path.join(filepath,
                                     "in.bgridlocal_defaultproc.python")
            else:
                self.parameters.lammps_compute_file = \
                    os.path.join(filepath, "in.bgrid.python")

        # Do the LAMMPS calculation.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (self.parameters.bispectrum_twojmax + 2) * \
                 (self.parameters.bispectrum_twojmax + 3) * (self.parameters.bispectrum_twojmax + 4)
        ncoeff = ncoeff // 24   # integer division
        self.fingerprint_length = ncols0+ncoeff

        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full SNAP array at the
        # end of this function.
        # This is not necessarily true for the parallel case.
        if self.parameters._configuration["mpi"]:
            nrows_local = extract_compute_np(lmp, "bgridlocal",
                                             lammps_constants.LMP_STYLE_LOCAL,
                                             lammps_constants.LMP_SIZE_ROWS)
            ncols_local = extract_compute_np(lmp, "bgridlocal",
                                             lammps_constants.LMP_STYLE_LOCAL,
                                             lammps_constants.LMP_SIZE_COLS)
            if ncols_local != self.fingerprint_length + 3:
                raise Exception("Inconsistent number of features.")

            snap_descriptors_np = \
                extract_compute_np(lmp, "bgridlocal",
                                   lammps_constants.LMP_STYLE_LOCAL, 2,
                                   array_shape=(nrows_local, ncols_local))

            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]

            # If I directly return the descriptors, this sometimes leads
            # to errors, because presumably the python garbage collection
            # deallocates memory too quickly. This copy is more memory
            # hungry, and we might have to tackle this later on, but
            # for now it works.
            return snap_descriptors_np.copy(), nrows_local

        else:
            # Extract data from LAMMPS calculation.
            snap_descriptors_np = \
                extract_compute_np(lmp, "bgrid", 0, 2,
                                   (nz, ny, nx, self.fingerprint_length))
            # switch from x-fastest to z-fastest order (swaps 0th and 2nd
            # dimension)
            snap_descriptors_np = snap_descriptors_np.transpose([2, 1, 0, 3])
            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            # If I directly return the descriptors, this sometimes leads
            # to errors, because presumably the python garbage collection
            # deallocates memory too quickly. This copy is more memory
            # hungry, and we might have to tackle this later on, but
            # for now it works.
            # I thought the transpose would take care of that, but apparently
            # it does not necessarily do that - so we have do go down
            # that route.
            if self.parameters.descriptors_contain_xyz:
                return snap_descriptors_np.copy(), nx*ny*nz
            else:
                return snap_descriptors_np[:, :, :, 3:].copy(), nx*ny*nz
