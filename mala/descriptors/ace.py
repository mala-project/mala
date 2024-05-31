"""ACE descriptor class."""

import os

import ase
import ase.io

from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance

from mala.common.parallelizer import printout
from mala.descriptors.lammps_utils import extract_compute_np
from mala.descriptors.descriptor import Descriptor


class ACE(Descriptor):
    """Class for calculation and parsing of bispectrum descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(ACE, self).__init__(parameters)

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "ACE"

    @property
    def feature_size(self):
        """Get the feature dimension of this data."""
        return self.fingerprint_length

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a bispectrum descriptor.

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
        if in_units == "None" or in_units is None:
            return array
        else:
            raise Exception("Unsupported unit for bispectrum descriptors.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a bispectrum descriptor.

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
        if out_units == "None" or out_units is None:
            return array
        else:
            raise Exception("Unsupported unit for bispectrum descriptors.")

    def _calculate(self, outdir, **kwargs):
        if self.parameters._configuration["lammps"]:
            if find_spec("lammps") is None:
                printout(
                    "No LAMMPS found for descriptor calculation, "
                    "terminating calculation."
                )
                sys.exit()
            else:
                return self.__calculate_lammps(outdir, **kwargs)

    def __calculate_lammps(self, outdir, **kwargs):
        """
        Perform bispectrum calculation using LAMMPS.

        Creates a LAMMPS instance with appropriate call parameters and uses
        it for the calculation.
        """
        # For version compatibility; older lammps versions (the serial version
        # we still use on some machines) have these constants as part of the
        # general LAMMPS import.
        from lammps import constants as lammps_constants

        use_fp64 = kwargs.get("use_fp64", False)
        keep_logs = kwargs.get("keep_logs", True)

        lammps_format = "lammps-data"
        self.lammps_temporary_input = os.path.join(
            outdir, "lammps_input_" + self.calculation_timestamp + ".tmp"
        )
        ase.io.write(
            self.lammps_temporary_input, self.atoms, format=lammps_format
        )

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Create LAMMPS instance.
        lammps_dict = {
            "ace_coeff_file": "coupling_coefficients.yace",
            "rcutfac": self.parameters.bispectrum_cutoff,
        }

        self.lammps_temporary_log = os.path.join(
            outdir,
            "lammps_acegrid_log_" + self.calculation_timestamp + ".tmp",
        )
        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("ace")[0]
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    self.parameters.lammps_compute_file = os.path.join(
                        filepath, "in.acegridlocal.python"
                    )
                else:
                    self.parameters.lammps_compute_file = os.path.join(
                        filepath, "in.acegridlocal_defaultproc.python"
                    )
            else:
                self.parameters.lammps_compute_file = os.path.join(
                    filepath, "in.acegrid.python"
                )

        # Do the LAMMPS calculation and clean up.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        # ncoeff = (
        #     (self.parameters.bispectrum_twojmax + 2)
        #     * (self.parameters.bispectrum_twojmax + 3)
        #     * (self.parameters.bispectrum_twojmax + 4)
        # )
        # ncoeff = ncoeff // 24  # integer division
        # self.fingerprint_length = ncols0 + ncoeff

        # Calculation for fingerprint length
        # TODO: I don't know if this is correct, should be checked by an ACE expert
        self.fingerprint_length = ncols0 + self._calculate_ace_fingerprint_length()

        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full bispectrum array at
        # the end of this function.
        # This is not necessarily true for the parallel case.
        if self.parameters._configuration["mpi"]:
            nrows_local = extract_compute_np(
                lmp,
                "mygridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_ROWS,
            )
            ncols_local = extract_compute_np(
                lmp,
                "mygridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_COLS,
            )
            if ncols_local != self.fingerprint_length + 3:
                raise Exception("Inconsistent number of features.")

            ace_descriptors_np = extract_compute_np(
                lmp,
                "mygridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                2,
                array_shape=(nrows_local, ncols_local),
                use_fp64=use_fp64,
            )
            self._clean_calculation(lmp, keep_logs)

            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            return ace_descriptors_np, nrows_local

        else:
            # Extract data from LAMMPS calculation.
            ace_descriptors_np = extract_compute_np(
                lmp,
                "mygrid",
                0,
                2,
                (nz, ny, nx, self.fingerprint_length),
                use_fp64=use_fp64,
            )
            self._clean_calculation(lmp, keep_logs)

            # switch from x-fastest to z-fastest order (swaps 0th and 2nd
            # dimension)
            ace_descriptors_np = ace_descriptors_np.transpose([2, 1, 0, 3])
            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            if self.parameters.descriptors_contain_xyz:
                return ace_descriptors_np, nx * ny * nz
            else:
                return ace_descriptors_np[:, :, :, 3:], nx * ny * nz

    def _calculate_ace_fingerprint_length(self):
        total_descriptors = 0
        ranks = self.parameters.ace_ranks
        lmax = self.parameters.ace_lmax
        nmax = self.parameters.ace_nmax
        lmin = self.parameters.ace_lmin

        for rank in range(len(ranks)):
            # Number of radial basis functions for the current rank
            num_radial_functions = nmax[rank]
        
            # Number of angular basis functions for the current rank
            num_angular_functions = (lmax[rank] - lmin[rank] + 1)
        
            # Total number of descriptors for the current rank
            rank_descriptors = num_radial_functions * num_angular_functions
        
            # Add to total descriptors
            total_descriptors += rank_descriptors
    
        return total_descriptors
