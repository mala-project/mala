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
import numpy as np

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.descriptors.atomic_density import AtomicDensity


class MinterpyDescriptors(Descriptor):
    """Class for calculation and parsing of Gaussian descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(MinterpyDescriptors, self).__init__(parameters)
        self.verbosity = parameters.verbosity

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "Minterpy"

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
            raise Exception("Unsupported unit for Minterpy descriptors.")

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
            raise Exception("Unsupported unit for Minterpy descriptors.")

    def _calculate(self, atoms, outdir, grid_dimensions, **kwargs):
        from lammps import lammps

        nx = grid_dimensions[0]
        ny = grid_dimensions[1]
        nz = grid_dimensions[2]

        # Check if we have to determine the optimal sigma value.
        if self.parameters.atomic_density_sigma is None:
            self.grid_dimensions = [nx, ny, nz]
            voxel = atoms.cell.copy()
            voxel[0] = voxel[0] / (self.grid_dimensions[0])
            voxel[1] = voxel[1] / (self.grid_dimensions[1])
            voxel[2] = voxel[2] / (self.grid_dimensions[2])
            self.parameters.atomic_density_sigma = AtomicDensity.\
                get_optimal_sigma(voxel)

        # Size of the local cube
        # self.parameters.minterpy_cutoff_cube_size

        # first grid point - for now just assume 0.0, 0.0, 0.0.
        # That is correct in 99% of the cases I am aware of, and once
        # we have a working implementation we can test and refine this.
        # first_grid_point = [0.0, 0.0, 0.0]

        # Get the unisolvent nodes.
        unisolvent_nodes = self._build_unisolvent_nodes()

        # Transform the unisolvent nodes into cartesian coordinates by assuming
        # small cubes around the grid points. Currently only works for cubic
        # cells.
        self.parameters.minterpy_point_list = []
        local_cube = atoms.cell.copy()
        local_cube[0] = local_cube[0] * (self.parameters.
                                         minterpy_cutoff_cube_size /
                                         local_cube[0][0])
        local_cube[1] = local_cube[1] * (self.parameters.
                                         minterpy_cutoff_cube_size /
                                         local_cube[0][0])
        local_cube[2] = local_cube[2] * (self.parameters.
                                         minterpy_cutoff_cube_size /
                                         local_cube[0][0])
        for i in range(np.shape(unisolvent_nodes)[0]):
            self.parameters.\
                minterpy_point_list.\
                append(np.matmul(local_cube, unisolvent_nodes[i]))

        # Array to hold descriptors.
        coord_length = 3 if self.parameters.descriptors_contain_xyz else 0
        minterpy_descriptors_np = \
            np.zeros([nx, ny, nz,
                      len(self.parameters.minterpy_point_list)+coord_length],
                     dtype=np.float64)
        self.fingerprint_length = \
            len(self.parameters.minterpy_point_list)+coord_length

        self.fingerprint_length = len(self.parameters.minterpy_point_list)
        # Perform one LAMMPS call for each point in the Minterpy point list.
        for idx, point in enumerate(self.parameters.minterpy_point_list):
            # Shift the atoms in negative direction of the point(s) we actually
            # want.
            atoms_copied = atoms.copy()
            atoms_copied.set_positions(atoms.get_positions()-np.array(point))

            # The rest is the stanfard LAMMPS atomic density stuff.
            lammps_format = "lammps-data"
            ase_out_path = os.path.join(outdir, "lammps_input.tmp")
            ase.io.write(ase_out_path, atoms_copied, format=lammps_format)

            # Create LAMMPS instance.
            lammps_dict = {}
            lammps_dict["sigma"] = self.parameters.atomic_density_sigma
            lammps_dict["rcutfac"] = self.parameters.atomic_density_cutoff
            lammps_dict["atom_config_fname"] = ase_out_path
            lmp = self._setup_lammps(nx, ny, nz, outdir, lammps_dict,
                                     log_file_name="lammps_mgrid_log.tmp")

            # For now the file is chosen automatically, because this is used
            # mostly under the hood anyway.
            filepath = __file__.split("minterpy")[0]
            if self.parameters._configuration["mpi"]:
                raise Exception("Minterpy descriptors cannot be calculated "
                                "in parallel yet.")
                # if self.parameters.use_z_splitting:
                #     runfile = os.path.join(filepath, "in.ggrid.python")
                # else:
                #     runfile = os.path.join(filepath, "in.ggrid_defaultproc.python")
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

            lmp.close()

            gaussian_descriptors_np = \
                gaussian_descriptors_np.reshape((grid_dimensions[2],
                                                 grid_dimensions[1],
                                                 grid_dimensions[0],
                                                 7))
            gaussian_descriptors_np = \
                gaussian_descriptors_np.transpose([2, 1, 0, 3])
            if self.parameters.descriptors_contain_xyz and idx == 0:
                minterpy_descriptors_np[:, :, :, 0:3] = \
                    gaussian_descriptors_np[:, :, :, 3:6].copy()

            minterpy_descriptors_np[:, :, :, coord_length+idx:coord_length+idx+1] = \
                gaussian_descriptors_np[:, :, :, 6:]

        return minterpy_descriptors_np, nx * ny * nz

    def _build_unisolvent_nodes(self, dimension=3):
        """
        Build the unisolvent nodes.

        Parameters
        ----------
        dimension : int
            Dimension of data (should always be 3 for 3D in our case).

        Returns
        -------
        nodes : numpy.array
            Two-dimensional array containing the nodes.
        """
        import minterpy as mp

        # Calculate the unisolvent nodes.
        mi = mp.MultiIndexSet.from_degree(spatial_dimension=dimension,
                                          poly_degree=self.parameters.minterpy_polynomial_degree,
                                          lp_degree=self.parameters.minterpy_lp_norm)
        unisolvent_nodes = mp.Grid(mi).unisolvent_nodes
        return unisolvent_nodes

