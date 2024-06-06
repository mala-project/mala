"""Gaussian descriptor class."""

import os

import ase
import ase.io
from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance

from mala.common.parallelizer import printout
from mala.descriptors.lammps_utils import extract_compute_np
from mala.descriptors.descriptor import Descriptor

# Empirical value for the Gaussian descriptor width, determined for an
# aluminium system. Reasonable values for sigma can and will be calculated
# automatically based on this value and the aluminium gridspacing
# for other systems as well.
optimal_sigma_aluminium = 0.2
reference_grid_spacing_aluminium = 0.08099000022712448


class AtomicDensity(Descriptor):
    """Class for calculation and parsing of Gaussian descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(AtomicDensity, self).__init__(parameters)
        self.verbosity = parameters.verbosity

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "AtomicDensity"

    @property
    def feature_size(self):
        """Get the feature dimension of this data."""
        return self.fingerprint_length

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
        if in_units == "None" or in_units is None:
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

    @staticmethod
    def get_optimal_sigma(voxel):
        """
        Calculate the optimal width of the Gaussians based on the grid voxel.

        Parameters
        ----------
        voxel : ase.Cell
            An ASE Cell object characterizing the voxel.

        Returns
        -------
        optimal_sigma : float
            The optimal sigma value.
        """
        return (
            np.max(voxel) / reference_grid_spacing_aluminium
        ) * optimal_sigma_aluminium

    def _calculate(self, outdir, **kwargs):
        if self.parameters._configuration["lammps"]:
            if find_spec("lammps") is None:
                printout(
                    "No LAMMPS found for descriptor calculation, "
                    "falling back to python."
                )
                return self.__calculate_python(**kwargs)
            else:
                return self.__calculate_lammps(outdir, **kwargs)
        else:
            return self.__calculate_python(**kwargs)

    def __calculate_lammps(self, outdir, **kwargs):
        """Perform actual Gaussian descriptor calculation."""
        # For version compatibility; older lammps versions (the serial version
        # we still use on some machines) have these constants as part of the
        # general LAMMPS import.
        from lammps import constants as lammps_constants

        use_fp64 = kwargs.get("use_fp64", False)
        return_directly = kwargs.get("return_directly", False)
        keep_logs = kwargs.get("keep_logs", False)

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

        # Check if we have to determine the optimal sigma value.
        if self.parameters.atomic_density_sigma is None:
            self.grid_dimensions = [nx, ny, nz]
            self.parameters.atomic_density_sigma = self.get_optimal_sigma(
                self.voxel
            )

        # Create LAMMPS instance.
        lammps_dict = {
            "sigma": self.parameters.atomic_density_sigma,
            "rcutfac": self.parameters.atomic_density_cutoff,
        }
        self.lammps_temporary_log = os.path.join(
            outdir,
            "lammps_ggrid_log_" + self.calculation_timestamp + ".tmp",
        )
        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # For now the file is chosen automatically, because this is used
        # mostly under the hood anyway.
        filepath = __file__.split("atomic_density")[0]
        if self.parameters._configuration["mpi"]:
            if self.parameters.use_z_splitting:
                self.parameters.lammps_compute_file = os.path.join(
                    filepath, "in.ggrid.python"
                )
            else:
                self.parameters.lammps_compute_file = os.path.join(
                    filepath, "in.ggrid_defaultproc.python"
                )
        else:
            self.parameters.lammps_compute_file = os.path.join(
                filepath, "in.ggrid_defaultproc.python"
            )

        # Do the LAMMPS calculation and clean up.
        lmp.file(self.parameters.lammps_compute_file)

        # Extract the data.
        nrows_ggrid = extract_compute_np(
            lmp,
            "ggrid",
            lammps_constants.LMP_STYLE_LOCAL,
            lammps_constants.LMP_SIZE_ROWS,
        )
        ncols_ggrid = extract_compute_np(
            lmp,
            "ggrid",
            lammps_constants.LMP_STYLE_LOCAL,
            lammps_constants.LMP_SIZE_COLS,
        )

        gaussian_descriptors_np = extract_compute_np(
            lmp,
            "ggrid",
            lammps_constants.LMP_STYLE_LOCAL,
            2,
            array_shape=(nrows_ggrid, ncols_ggrid),
            use_fp64=use_fp64,
        )
        self._clean_calculation(lmp, keep_logs)

        # In comparison to SNAP, the atomic density always returns
        # in the "local mode". Thus we have to make some slight adjustments
        # if we operate without MPI.
        self.grid_dimensions = [nx, ny, nz]
        if self.parameters._configuration["mpi"]:
            if return_directly:
                return gaussian_descriptors_np
            else:
                self.fingerprint_length = 4
                return gaussian_descriptors_np, nrows_ggrid
        else:
            # Since the atomic density may be directly fed back into QE
            # during the total energy calculation, we may have to return
            # the descriptors, even in serial mode, without any further
            # reordering.
            if return_directly:
                return gaussian_descriptors_np
            else:
                # Here, we want to do something else with the atomic density,
                # and thus have to properly reorder it.
                # We have to switch from x fastest to z fastest reordering.
                gaussian_descriptors_np = gaussian_descriptors_np.reshape(
                    (
                        self.grid_dimensions[2],
                        self.grid_dimensions[1],
                        self.grid_dimensions[0],
                        7,
                    )
                )
                gaussian_descriptors_np = gaussian_descriptors_np.transpose(
                    [2, 1, 0, 3]
                )
                if self.parameters.descriptors_contain_xyz:
                    self.fingerprint_length = 4
                    return gaussian_descriptors_np[:, :, :, 3:], nx * ny * nz
                else:
                    self.fingerprint_length = 1
                    return gaussian_descriptors_np[:, :, :, 6:], nx * ny * nz

    def __calculate_python(self, **kwargs):
        """
        Perform Gaussian descriptor calculation using python.

        The code used to this end was adapted from the LAMMPS implementation.
        It serves as a fallback option whereever LAMMPS is not available.
        This may be useful, e.g., to students or people getting started with
        MALA who just want to look around. It is not intended for production
        calculations.
        Compared to the LAMMPS implementation, this implementation has quite a
        few limitations. Namely

            - It is roughly an order of magnitude slower for small systems
              and doesn't scale too great
            - It only works for ONE chemical element
            - It has no MPI or GPU support
        """
        printout(
            "Using python for descriptor calculation. "
            "The resulting calculation will be slow for "
            "large systems."
        )

        gaussian_descriptors_np = np.zeros(
            (
                self.grid_dimensions[0],
                self.grid_dimensions[1],
                self.grid_dimensions[2],
                4,
            ),
            dtype=np.float64,
        )

        # Construct the hyperparameters to calculate the Gaussians.
        # This follows the implementation in the LAMMPS code.
        if self.parameters.atomic_density_sigma is None:
            self.parameters.atomic_density_sigma = self.get_optimal_sigma(
                self.voxel
            )
        cutoff_squared = (
            self.parameters.atomic_density_cutoff
            * self.parameters.atomic_density_cutoff
        )
        prefactor = 1.0 / (
            np.power(
                self.parameters.atomic_density_sigma * np.sqrt(2 * np.pi), 3
            )
        )
        argumentfactor = 1.0 / (
            2.0
            * self.parameters.atomic_density_sigma
            * self.parameters.atomic_density_sigma
        )

        # Create a list of all potentially relevant atoms.
        all_atoms = self._setup_atom_list()

        # I think this nested for-loop could probably be optimized if instead
        # the density matrix is used on the entire grid. That would be VERY
        # memory-intensive. Since the goal of such an optimization would be
        # to use this implementation at potentially larger length-scales,
        # one would have to investigate that this is OK memory-wise.
        # I haven't optimized it yet for the smaller scales since there
        # the performance was already good enough.
        for i in range(0, self.grid_dimensions[0]):
            for j in range(0, self.grid_dimensions[1]):
                for k in range(0, self.grid_dimensions[2]):
                    # Compute the grid.
                    gaussian_descriptors_np[i, j, k, 0:3] = (
                        self._grid_to_coord([i, j, k])
                    )

                    # Compute the Gaussian descriptors.
                    dm = np.squeeze(
                        distance.cdist(
                            [gaussian_descriptors_np[i, j, k, 0:3]], all_atoms
                        )
                    )
                    dm = dm * dm
                    dm_cutoff = dm[np.argwhere(dm < cutoff_squared)]
                    gaussian_descriptors_np[i, j, k, 3] += np.sum(
                        prefactor * np.exp(-dm_cutoff * argumentfactor)
                    )

        if self.parameters.descriptors_contain_xyz:
            self.fingerprint_length = 4
            return gaussian_descriptors_np, np.prod(self.grid_dimensions)
        else:
            self.fingerprint_length = 1
            return gaussian_descriptors_np[:, :, :, 3:], np.prod(
                self.grid_dimensions
            )
