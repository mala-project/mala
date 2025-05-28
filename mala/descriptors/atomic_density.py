"""Gaussian descriptor class."""

import os

import ase
import ase.io
from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance

from mala.common.parallelizer import printout, parallel_warn
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

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "AtomicDensity"

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a Gaussian descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.ndarray
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.ndarray
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
        array : numpy.ndarray
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.ndarray
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
        """
        Perform Gaussian descriptor calculation.

        This function is the main entry point for the calculation of the
        Gaussian descriptors. It will decide whether to use LAMMPS or
        a python implementation based on the availability of LAMMPS (and
        user specifications).

        Parameters
        ----------
        outdir : string
            Path to the output directory.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        gaussian_descriptors_np : numpy.ndarray
            The calculated Gaussian descriptors.
        """
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

    def _read_feature_dimension_from_json(self, json_dict):
        """
        Read the feature dimension from a saved JSON file.

        Always returns (number of coordinate dimensions xyz = 3) +
        (number of species), so for a single species system 4.

        Parameters
        ----------
        json_dict : dict
            Dictionary containing info loaded from the JSON file.
        """
        return 4

    def __calculate_lammps(self, outdir, **kwargs):
        """
        Perform actual Gaussian descriptor calculation using LAMMPS.

        Creates a LAMMPS instance with appropriate call parameters and uses
        it for the calculation.

        Parameters
        ----------
        outdir : string
            Path to the output directory.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        gaussian_descriptors_np : numpy.ndarray
            The calculated Gaussian descriptors.

        """
        # For version compatibility; older lammps versions (the serial version
        # we still use on some machines) have these constants as part of the
        # general LAMMPS import.
        from lammps import constants as lammps_constants

        use_fp64 = kwargs.get("use_fp64", False)
        return_directly = kwargs.get("return_directly", False)
        keep_logs = kwargs.get("keep_logs", False)

        lammps_format = "lammps-data"
        self.setup_lammps_tmp_files("ggrid", outdir)

        ase.io.write(
            self._lammps_temporary_input, self._atoms, format=lammps_format
        )

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Check if we have to determine the optimal sigma value.
        if self.parameters.atomic_density_sigma is None:
            self.grid_dimensions = [nx, ny, nz]
            self.parameters.atomic_density_sigma = self.get_optimal_sigma(
                self._voxel
            )

        # Create LAMMPS instance.
        lammps_dict = {
            "rcutfac": self.parameters.atomic_density_cutoff,
        }
        for i in range(len(set(self._atoms.numbers))):
            lammps_dict["sigma" + str(i + 1)] = (
                self.parameters.atomic_density_sigma
            )

        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # For now the file is chosen automatically, because this is used
        # mostly under the hood anyway.

        if self.parameters.custom_lammps_compute_file != "":
            lammps_compute_file = self.parameters.custom_lammps_compute_file
        else:
            filepath = os.path.join(os.path.dirname(__file__), "inputfiles")
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    lammps_compute_file = os.path.join(
                        filepath,
                        "in.ggrid_n{0}.python".format(
                            len(set(self._atoms.numbers))
                        ),
                    )
                else:
                    lammps_compute_file = os.path.join(
                        filepath,
                        "in.ggrid_defaultproc_n{0}.python".format(
                            len(set(self._atoms.numbers))
                        ),
                    )
            else:
                lammps_compute_file = os.path.join(
                    filepath,
                    "in.ggrid_defaultproc_n{0}.python".format(
                        len(set(self._atoms.numbers))
                    ),
                )

        # Do the LAMMPS calculation and clean up.
        lmp.file(lammps_compute_file)

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

        if len(set(self._atoms.numbers)) > 1:
            parallel_warn(
                "Atomic density formula and multielement system detected: "
                "Quantum ESPRESSO has a different internal order "
                "for structure factors and atomic positions. "
                "MALA recovers the correct ordering for multielement "
                "systems, but the algorithm to do so is still "
                "experimental. Please test on a small system before "
                "using the atomic density formula at scale."
            )
            symbols, indices = np.unique(
                [atom.symbol for atom in self._atoms], return_index=True
            )
            permutation = np.concatenate(
                ([0, 1, 2, 3, 4, 5], np.argsort(indices) + 6)
            )
            gaussian_descriptors_np = gaussian_descriptors_np[:, permutation]

        # In comparison to bispectrum, the atomic density always returns
        # in the "local mode". Thus we have to make some slight adjustments
        # if we operate without MPI.
        self.grid_dimensions = [nx, ny, nz]
        if self.parameters._configuration["mpi"]:
            if return_directly:
                return gaussian_descriptors_np
            else:
                self.feature_size = 4
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
                        6 + len(set(self._atoms.numbers)),
                    )
                )
                gaussian_descriptors_np = gaussian_descriptors_np.transpose(
                    [2, 1, 0, 3]
                )
                if self.parameters.descriptors_contain_xyz:
                    self.feature_size = 4
                    return gaussian_descriptors_np[:, :, :, 3:], nx * ny * nz
                else:
                    self.feature_size = 1
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

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        gaussian_descriptors_np : numpy.ndarray
            The calculated Gaussian descriptors.
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
                self._voxel
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
            self.feature_size = 4
            return gaussian_descriptors_np, np.prod(self.grid_dimensions)
        else:
            self.feature_size = 1
            return gaussian_descriptors_np[:, :, :, 3:], np.prod(
                self.grid_dimensions
            )
