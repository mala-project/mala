"""Bispectrum descriptor class."""
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
from scipy.spatial import distance

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor


class Bispectrum(Descriptor):
    """Class for calculation and parsing of bispectrum descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(Bispectrum, self).__init__(parameters)

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "Bispectrum"

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
            return self.__calculate_lammps(outdir, **kwargs)
        else:
            return self.__calculate_python(**kwargs)

    def __calculate_lammps(self, outdir, **kwargs):
        """Perform actual bispectrum calculation."""
        use_fp64 = kwargs.get("use_fp64", False)

        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, self.atoms, format=lammps_format)

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Create LAMMPS instance.
        lammps_dict = {}
        lammps_dict["twojmax"] = self.parameters.bispectrum_twojmax
        lammps_dict["rcutfac"] = self.parameters.bispectrum_cutoff
        lammps_dict["atom_config_fname"] = ase_out_path
        lmp = self._setup_lammps(nx, ny, nz, outdir, lammps_dict,
                                 log_file_name="lammps_bgrid_log.tmp")

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
        # In the serial case we can expect to have a full bispectrum array at
        # the end of this function.
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
                                   array_shape=(nrows_local, ncols_local),
                                   use_fp64=use_fp64)
            lmp.close()

            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            return snap_descriptors_np, nrows_local

        else:
            # Extract data from LAMMPS calculation.
            snap_descriptors_np = \
                extract_compute_np(lmp, "bgrid", 0, 2,
                                   (nz, ny, nx, self.fingerprint_length),
                                   use_fp64=use_fp64)
            lmp.close()

            # switch from x-fastest to z-fastest order (swaps 0th and 2nd
            # dimension)
            snap_descriptors_np = snap_descriptors_np.transpose([2, 1, 0, 3])
            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            if self.parameters.descriptors_contain_xyz:
                return snap_descriptors_np, nx*ny*nz
            else:
                return snap_descriptors_np[:, :, :, 3:], nx*ny*nz

    def __calculate_python(self, **kwargs):
        ncoeff = (self.parameters.bispectrum_twojmax + 2) * \
                 (self.parameters.bispectrum_twojmax + 3) * (self.parameters.bispectrum_twojmax + 4)
        ncoeff = ncoeff // 24   # integer division
        self.fingerprint_length = ncoeff + 3
        bispectrum_np = np.zeros((self.grid_dimensions[0],
                                  self.grid_dimensions[1],
                                  self.grid_dimensions[2],
                                  self.fingerprint_length),
                                 dtype=np.float64)
        cutoff_squared = self.parameters.bispectrum_cutoff * \
            self.parameters.bispectrum_cutoff

        all_atoms = self._setup_atom_list()

        # These are technically hyperparameters. We currently simply set them
        # to set values for everything.
        rmin0 = 0.0
        rfac0 = 0.99363

        self.__init_index_arrays()

        for x in range(0, self.grid_dimensions[0]):
            for y in range(0, self.grid_dimensions[1]):
                for z in range(0, self.grid_dimensions[2]):
                    # Compute the grid.
                    bispectrum_np[x, y, z, 0:3] = \
                        self._grid_to_coord([x, y, z])

                    # Compute the bispectrum descriptors.
                    distances = np.squeeze(distance.cdist(
                        [bispectrum_np[x, y, z, 0:3]],
                        all_atoms))
                    distances_squared = distances*distances
                    distances_squared_cutoff = distances_squared[np.argwhere(distances_squared < cutoff_squared)]
                    distances_cutoff = np.abs(distances[np.argwhere(distances < self.parameters.bispectrum_cutoff)])
                    atoms_cutoff = np.squeeze(all_atoms[np.argwhere(distances < self.parameters.bispectrum_cutoff), :])
                    nr_atoms = np.shape(atoms_cutoff)[0]

                    ulisttot_r, ulisttot_i = \
                        self.__compute_ui(nr_atoms, atoms_cutoff,
                                          distances_cutoff,
                                          distances_squared_cutoff, rmin0,
                                          rfac0)
                    print("Got Ui")

                    #
                    # gaussian_descriptors_np[i, j, k, 3] += \
                    #     np.sum(prefactor*np.exp(-dm_cutoff*argumentfactor))

        return bispectrum_np, np.prod(self.grid_dimensions)

    def __init_index_arrays(self):
        # TODO: Declare these in constructor!
        idxu_count = 0
        self.idxu_block = np.zeros(self.parameters.bispectrum_twojmax + 1)
        for j in range(0, self.parameters.bispectrum_twojmax + 1):
            self.idxu_block[j] = idxu_count
            for mb in range(j + 1):
                for ma in range(j + 1):
                    idxu_count += 1
        self.idxu_max = idxu_count

        self.rootpqarray = np.zeros((self.parameters.bispectrum_twojmax + 2,
                                    self.parameters.bispectrum_twojmax + 2))
        for p in range(1, self.parameters.bispectrum_twojmax + 1):
            for q in range(1,
                           self.parameters.bispectrum_twojmax + 1):
                self.rootpqarray[p, q] = np.sqrt(p / q)

    def __compute_ui(self, nr_atoms, atoms_cutoff, distances_cutoff,
                     distances_squared_cutoff, rmin0, rfac0):
        # Precompute and prepare ui stuff
        theta0 = (distances_cutoff - rmin0) * rfac0 * np.pi / (
                    self.parameters.bispectrum_cutoff - rmin0)
        z0 = distances_cutoff / np.tan(theta0)

        ulist_r_ij = np.zeros((nr_atoms, self.idxu_max)) + 1.0
        ulist_i_ij = np.zeros((nr_atoms, self.idxu_max))
        ulisttot_r = np.zeros(self.idxu_max)
        ulisttot_i = np.zeros(self.idxu_max)
        r0inv = 1.0 / np.sqrt(distances_cutoff)

        for a in range(nr_atoms):
            # This encapsulates the compute_uarray function

            # Cayley-Klein parameters for unit quaternion.
            a_r = r0inv[a] * z0[a]
            a_i = -r0inv[a] * atoms_cutoff[a, 2]
            b_r = r0inv[a] * atoms_cutoff[a, 1]
            b_i = -r0inv[a] * atoms_cutoff[a, 0]

            for j in range(1, self.parameters.bispectrum_twojmax + 1):
                jju = int(self.idxu_block[j])
                jjup = int(self.idxu_block[j - 1])

                for mb in range(0, j // 2 + 1):
                    ulist_r_ij[a, jju] = 0.0
                    ulist_i_ij[a, jju] = 0.0
                    for ma in range(0, j):
                        rootpq = self.rootpqarray[j - ma][j - mb]
                        ulist_r_ij[a, jju] += rootpq * (
                                a_r * ulist_r_ij[a, jjup] + a_i *
                                ulist_i_ij[a, jjup])
                        ulist_i_ij[a, jju] += rootpq * (
                                a_r * ulist_i_ij[a, jjup] - a_i *
                                ulist_r_ij[a, jjup])
                        rootpq = self.rootpqarray[ma + 1][j - mb]
                        ulist_r_ij[a, jju + 1] = -rootpq * (
                                b_r * ulist_r_ij[a, jjup] + b_i *
                                ulist_i_ij[a, jjup])
                        ulist_i_ij[a, jju + 1] = -rootpq * (
                                b_r * ulist_i_ij[a, jjup] - b_i *
                                ulist_r_ij[a, jjup])
                        jju += 1
                        jjup += 1
                    jju += 1

                jju = int(self.idxu_block[j])
                jjup = int(jju + (j + 1) * (j + 1) - 1)
                mbpar = 1
                for mb in range(0, j // 2 + 1):
                    mapar = mbpar
                    for ma in range(0, j + 1):
                        if mapar == 1:
                            ulist_r_ij[a, jjup] = ulist_r_ij[a, jju]
                            ulist_i_ij[a, jjup] = -ulist_i_ij[a, jju]
                        else:
                            ulist_r_ij[a, jjup] = -ulist_r_ij[a, jju]
                            ulist_i_ij[a, jjup] = ulist_i_ij[a, jju]
                        mapar = -mapar
                        jju += 1
                        jjup -= 1
                    mbpar = -mbpar

            # This emulates add_uarraytot.
            # First, we compute sfac.
            if self.parameters.bispectrum_switchflag == 0:
                sfac = 1.0
            elif distances_cutoff[a] <= rmin0:
                sfac = 1.0
            elif distances_cutoff[a] > self.parameters.bispectrum_cutoff:
                sfac = 0.0
            else:
                rcutfac = np.pi / (self.parameters.bispectrum_cutoff - rmin0)
                sfac = 0.5 * (np.cos((distances_cutoff[a] - rmin0) * rcutfac)
                              + 1.0)

            # sfac technically has to be weighted according to the chemical
            # species. But this is a minimal implementation only for a single
            # chemical species, so I am ommitting this for now. It would
            # look something like
            # sfac *= weights[a]
            # Further, some things have to be calculated if
            # switch_inner_flag is true. If I understand correctly, it
            # essentially never is in our case. So I am ommitting this
            # (along with some other similar lines) here for now.
            # If this becomes relevant later, we of course have to
            # add it.

            # Now use sfac for computations.
            for j in range(self.parameters.bispectrum_twojmax + 1):
                jju = int(self.idxu_block[j])
                for mb in range(j + 1):
                    for ma in range(j + 1):
                        ulisttot_r[jju] += sfac * ulist_r_ij[a,
                            jju]
                        ulisttot_i[jju] += sfac * ulist_i_ij[a,
                            jju]
                        jju += 1

        return ulisttot_r, ulisttot_i

