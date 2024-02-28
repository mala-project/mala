"""Gaussian descriptor class."""
import os

import ase
import ase.io
from ase.neighborlist import NeighborList
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
from scipy.spatial import distance

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
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
        return (np.max(voxel) / reference_grid_spacing_aluminium) * \
               optimal_sigma_aluminium

    def _calculate(self, outdir, **kwargs):
        if self.parameters._configuration["lammps"]:
            return self.__calculate_lammps(outdir, **kwargs)
        else:
            return self.__calculate_python(**kwargs)

    def __calculate_lammps(self, outdir, **kwargs):
        """Perform actual Gaussian descriptor calculation."""
        use_fp64 = kwargs.get("use_fp64", False)
        return_directly = kwargs.get("return_directly", False)

        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, self.atoms, format=lammps_format)

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Check if we have to determine the optimal sigma value.
        if self.parameters.atomic_density_sigma is None:
            self.grid_dimensions = [nx, ny, nz]
            self.parameters.atomic_density_sigma = self.\
                get_optimal_sigma(self.voxel)

        # Create LAMMPS instance.
        lammps_dict = {}
        lammps_dict["sigma"] = self.parameters.atomic_density_sigma
        lammps_dict["rcutfac"] = self.parameters.atomic_density_cutoff
        lammps_dict["atom_config_fname"] = ase_out_path
        lmp = self._setup_lammps(nx, ny, nz, outdir, lammps_dict,
                                 log_file_name="lammps_ggrid_log.tmp")

        # For now the file is chosen automatically, because this is used
        # mostly under the hood anyway.
        filepath = __file__.split("atomic_density")[0]
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
                               array_shape=(nrows_ggrid, ncols_ggrid),
                               use_fp64=use_fp64)
        lmp.close()

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
                gaussian_descriptors_np = \
                    gaussian_descriptors_np.reshape((self.grid_dimensions[2],
                                                     self.grid_dimensions[1],
                                                     self.grid_dimensions[0],
                                                     7))
                gaussian_descriptors_np = \
                    gaussian_descriptors_np.transpose([2, 1, 0, 3])
                if self.parameters.descriptors_contain_xyz:
                    self.fingerprint_length = 4
                    return gaussian_descriptors_np[:, :, :, 3:], \
                           nx*ny*nz
                else:
                    self.fingerprint_length = 1
                    return gaussian_descriptors_np[:, :, :, 6:], \
                           nx*ny*nz

    def __calculate_python(self, **kwargs):
        gaussian_descriptors_np = np.zeros((self.grid_dimensions[0],
                                            self.grid_dimensions[1],
                                            self.grid_dimensions[2], 4),
                                           dtype=np.float64)

        # Construct the hyperparameters to calculate the Gaussians.
        # This follows the implementation in the LAMMPS code.
        if self.parameters.atomic_density_sigma is None:
            self.parameters.atomic_density_sigma = self.\
                get_optimal_sigma(self.voxel)
        cutoff_squared = self.parameters.atomic_density_cutoff * \
            self.parameters.atomic_density_cutoff
        prefactor = 1.0 / (np.power(self.parameters.atomic_density_sigma *
                                    np.sqrt(2*np.pi),3))
        argumentfactor = 1.0 / (2.0 * self.parameters.atomic_density_sigma *
                                self.parameters.atomic_density_sigma)

        # If periodic boundary conditions are used, which is usually the case
        # for MALA simulation, one has to compute the atomic density by also
        # incorporating atoms from neighboring cells.
        # To do this efficiently, here we first check which cells have to be
        # included in the calculation.
        # For this we simply take the edges of the simulation cell and
        # construct neighor lists with the selected cutoff radius.
        # Each neighboring cell which is included in the neighbor list for
        # one of the edges will be considered for the calculation of the
        # Gaussians.
        # This approach may become inefficient for larger cells, in which
        # case this python based implementation should not be used
        # at any rate.
        all_index_offset_pairs = None
        if np.any(self.atoms.pbc):
            edges = [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
            for edge in edges:
                edge_point = self.__grid_to_coord(edge)
                neighborlist = ase.neighborlist.NeighborList(
                    np.zeros(len(self.atoms)+1) +
                    [self.parameters.atomic_density_cutoff],
                    bothways=True,
                    self_interaction=False,
                    primitive=ase.neighborlist.NewPrimitiveNeighborList)

                atoms_with_grid_point = self.atoms.copy()

                # Construct a ghost atom representing the grid point.
                atoms_with_grid_point.append(ase.Atom("H", edge_point))
                neighborlist.update(atoms_with_grid_point)
                indices, offsets = neighborlist.get_neighbors(len(self.atoms))

                offsets_without_grid = np.squeeze(offsets[np.argwhere(indices < len(self.atoms))])
                indices_without_grid = indices[np.argwhere(indices < len(self.atoms))]
                if all_index_offset_pairs is None:
                    all_index_offset_pairs = np.concatenate((indices_without_grid, offsets_without_grid), axis=1)
                else:
                    all_index_offset_pairs = np.concatenate((all_index_offset_pairs, np.concatenate((indices_without_grid, offsets_without_grid), axis=1)))
            all_index_offset_pairs_unique = np.unique(all_index_offset_pairs, axis=0)
            all_atoms = np.zeros((np.shape(all_index_offset_pairs_unique)[0], 3))
            for a in range(np.shape(all_index_offset_pairs_unique)[0]):
                all_atoms[a] = self.atoms.positions[all_index_offset_pairs_unique[a,0]] + all_index_offset_pairs_unique[a,1:] @ self.atoms.get_cell()
        else:
            # If no PBC are used, only consider a single cell.
            all_atoms = self.atoms.positions

        for i in range(0, self.grid_dimensions[0]):
            for j in range(0, self.grid_dimensions[1]):
                for k in range(0, self.grid_dimensions[2]):
                    # Compute the grid.
                    gaussian_descriptors_np[i, j, k, 0:3] = \
                        self.__grid_to_coord([i, j, k])

                    # Compute the Gaussian descriptors.
                    dm = np.squeeze(distance.cdist(
                        [gaussian_descriptors_np[i, j, k, 0:3]],
                        all_atoms))
                    dm = dm*dm
                    dm_cutoff = dm[np.argwhere(dm < cutoff_squared)]
                    gaussian_descriptors_np[i, j, k, 3] += \
                        np.sum(prefactor*np.exp(-dm_cutoff*argumentfactor))

        return gaussian_descriptors_np, np.prod(self.grid_dimensions)

    def __grid_to_coord(self, gridpoint):
        # Convert grid indices to real space grid point.
        i = gridpoint[0]
        j = gridpoint[1]
        k = gridpoint[2]
        # Orthorhombic cells and triclinic ones have
        # to be treated differently, see domain.cpp

        if self.atoms.cell.orthorhombic:
            return np.diag(self.voxel) * [i, j, k]
        else:
            ret = [0, 0, 0]
            ret[0] = i / self.grid_dimensions[0] * self.atoms.cell[0, 0] + \
                j / self.grid_dimensions[1] * self.atoms.cell[1, 0] + \
                k / self.grid_dimensions[2] * self.atoms.cell[2, 0]
            ret[1] = j / self.grid_dimensions[1] * self.atoms.cell[1, 1] + \
                k / self.grid_dimensions[2] * self.atoms.cell[1, 2]
            ret[2] = k / self.grid_dimensions[2] * self.atoms.cell[2, 2]
            return np.array(ret)
