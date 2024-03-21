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
        self.rmin0 = 0.0
        self.rfac0 = 0.99363
        self.bzero_flag = False
        self.wselfall_flag = False
        self.bnorm_flag = False
        self.quadraticflag = False

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

                    printer = False
                    if x == 0 and y == 0 and z == 1:
                        printer = True

                    ulisttot_r, ulisttot_i = \
                        self.__compute_ui(nr_atoms, atoms_cutoff,
                                          distances_cutoff,
                                          distances_squared_cutoff, bispectrum_np[x,y,z,0:3],
                                          printer)
                    if x == 0 and y == 0 and z == 1:
                        print("ulisttot_r i", ulisttot_r[0], ulisttot_i[0])
                        print("ulisttot_r i", ulisttot_r[1], ulisttot_i[1])
                        print("idxu_block", self.idxu_block[5])

                    zlist_r, zlist_i = \
                        self.__compute_zi(ulisttot_r, ulisttot_i)
                    blist = \
                        self.__compute_bi(ulisttot_r, ulisttot_i, zlist_r, zlist_i)

                    bispectrum_np[x, y, z, 3:] = blist
                    if x == 0 and y == 0 and z == 1:
                        print("BISPECTRUM", bispectrum_np[x, y, z, :])
                        exit()

                    # This will basically never be used. We don't really
                    # need to optimize it for now.
                    if self.quadraticflag:
                        ncount = ncoeff
                        for icoeff in range(ncoeff):
                            bveci = blist[icoeff]
                            bispectrum_np[x, y, z, 3 + ncount] = 0.5 * bveci * bveci
                            ncount += 1
                            for jcoeff in range(icoeff + 1, ncoeff):
                                bispectrum_np[x, y, z, 3 + ncount] = bveci * \
                                                                     blist[
                                                                         jcoeff]
                                ncount += 1

                    #
                    # gaussian_descriptors_np[i, j, k, 3] += \
                    #     np.sum(prefactor*np.exp(-dm_cutoff*argumentfactor))

        return bispectrum_np, np.prod(self.grid_dimensions)

    class ZIndices:

        def __init__(self):
            self.j1 = 0
            self.j2 = 0
            self.j = 0
            self.ma1min = 0
            self.ma2max = 0
            self.mb1min = 0
            self.mb2max = 0
            self.na = 0
            self.nb = 0
            self.jju = 0

    class BIndices:

        def __init__(self):
            self.j1 = 0
            self.j2 = 0
            self.j = 0

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

        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            idxz_count += 1
        self.idxz_max = idxz_count
        self.idxz = [self.ZIndices()]*self.idxz_max
        self.idxz_block = np.zeros((self.parameters.bispectrum_twojmax + 1,
                                    self.parameters.bispectrum_twojmax + 1,
                                    self.parameters.bispectrum_twojmax + 1))

        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    self.idxz_block[j1][j2][j] = idxz_count

                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            self.idxz[idxz_count].j1 = j1
                            self.idxz[idxz_count].j2 = j2
                            self.idxz[idxz_count].j = j
                            self.idxz[idxz_count].ma1min = max(0, (
                                        2 * ma - j - j2 + j1) // 2)
                            self.idxz[idxz_count].ma2max = (2 * ma - j - (2 * self.idxz[
                                idxz_count].ma1min - j1) + j2) // 2
                            self.idxz[idxz_count].na = min(j1, (
                                        2 * ma - j + j2 + j1) // 2) - self.idxz[
                                                      idxz_count].ma1min + 1
                            self.idxz[idxz_count].mb1min = max(0, (
                                        2 * mb - j - j2 + j1) // 2)
                            self.idxz[idxz_count].mb2max = (2 * mb - j - (2 * self.idxz[
                                idxz_count].mb1min - j1) + j2) // 2
                            self.idxz[idxz_count].nb = min(j1, (
                                        2 * mb - j + j2 + j1) // 2) - self.idxz[
                                                      idxz_count].mb1min + 1

                            jju = self.idxu_block[j] + (j + 1) * mb + ma
                            self.idxz[idxz_count].jju = jju
                            idxz_count += 1

        self.idxcg_block = np.zeros((self.parameters.bispectrum_twojmax + 1,
                                     self.parameters.bispectrum_twojmax + 1,
                                     self.parameters.bispectrum_twojmax + 1))
        idxcg_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    self.idxcg_block[j1][j2][j] = idxcg_count
                    for m1 in range(j1 + 1):
                        for m2 in range(j2 + 1):
                            idxcg_count += 1
        self.idxcg_max = idxcg_count
        self.cglist = np.zeros(self.idxcg_max)

        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    if j >= j1:
                        idxb_count += 1
        self.idxb_max = idxb_count
        self.idxb = [self.BIndices()]*self.idxb_max
        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        self.idxb[idxb_count].j1 = j1
                        self.idxb[idxb_count].j2 = j2
                        self.idxb[idxb_count].j = j
                        idxb_count += 1
        self.idxb_block = np.zeros((self.parameters.bispectrum_twojmax + 1,
                                    self.parameters.bispectrum_twojmax + 1,
                                    self.parameters.bispectrum_twojmax + 1))

        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    if j >= j1:
                        self.idxb_block[j1][j2][j] = idxb_count
                        idxb_count += 1

    def __compute_ui(self, nr_atoms, atoms_cutoff, distances_cutoff,
                     distances_squared_cutoff, grid, printer=False):
        # Precompute and prepare ui stuff
        theta0 = (distances_cutoff - self.rmin0) * self.rfac0 * np.pi / (
                    self.parameters.bispectrum_cutoff - self.rmin0)
        z0 = distances_cutoff / np.tan(theta0)

        ulist_r_ij = np.zeros((nr_atoms, self.idxu_max)) + 1.0
        ulist_i_ij = np.zeros((nr_atoms, self.idxu_max))
        ulisttot_r = np.zeros(self.idxu_max) + 1.0 # Actually probably wself
        ulisttot_i = np.zeros(self.idxu_max)
        r0inv = 1.0 / np.sqrt(distances_cutoff*distances_cutoff + z0*z0)

        for a in range(nr_atoms):
            # This encapsulates the compute_uarray function

            # Cayley-Klein parameters for unit quaternion.
            a_r = r0inv[a] * z0[a]
            a_i = -r0inv[a] * (grid[2]-atoms_cutoff[a, 2])
            b_r = r0inv[a] * (grid[1]-atoms_cutoff[a, 1])
            b_i = -r0inv[a] * (grid[0]-atoms_cutoff[a, 0])
            if printer:
                print(distances_cutoff[a][0], grid[0]-atoms_cutoff[a, 0], grid[1]-atoms_cutoff[a, 1], grid[2]-atoms_cutoff[a, 2],
                      a_r[0], a_i[0], b_r[0], b_i[0], r0inv[a][0], z0[a][0])

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
            elif distances_cutoff[a] <= self.rmin0:
                sfac = 1.0
            elif distances_cutoff[a] > self.parameters.bispectrum_cutoff:
                sfac = 0.0
            else:
                rcutfac = np.pi / (self.parameters.bispectrum_cutoff -
                                   self.rmin0)
                sfac = 0.5 * (np.cos((distances_cutoff[a] - self.rmin0) * rcutfac)
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

    def __compute_zi(self, ulisttot_r, ulisttot_i):
        # For now set the number of elements to 1.
        # This also has some implications for the rest of the function.
        # This currently really only works for one element.
        number_elements = 1
        number_element_pairs = number_elements*number_elements
        zlist_r = np.zeros((number_element_pairs*self.idxz_max))
        zlist_i = np.zeros((number_element_pairs*self.idxz_max))
        idouble = 0
        for elem1 in range(0, number_elements):
            for elem2 in range(0, number_elements):
                for jjz in range(self.idxz_max):
                    j1 = self.idxz[jjz].j1
                    j2 = self.idxz[jjz].j2
                    j = self.idxz[jjz].j
                    ma1min = self.idxz[jjz].ma1min
                    ma2max = self.idxz[jjz].ma2max
                    na = self.idxz[jjz].na
                    mb1min = self.idxz[jjz].mb1min
                    mb2max = self.idxz[jjz].mb2max
                    nb = self.idxz[jjz].nb
                    cgblock = self.cglist + self.idxcg_block[j1][j2][j]
                    zlist_r[jjz] = 0.0
                    zlist_i[jjz] = 0.0
                    jju1 = int(self.idxu_block[j1] + (j1 + 1) * mb1min)
                    jju2 = int(self.idxu_block[j2] + (j2 + 1) * mb2max)
                    icgb = mb1min * (j2 + 1) + mb2max
                    for ib in range(nb):
                        suma1_r = 0.0
                        suma1_i = 0.0
                        u1_r = ulisttot_r[elem1 * self.idxu_max + jju1:]
                        u1_i = ulisttot_i[elem1 * self.idxu_max + jju1:]
                        u2_r = ulisttot_r[elem2 * self.idxu_max + jju2:]
                        u2_i = ulisttot_i[elem2 * self.idxu_max + jju2:]
                        ma1 = ma1min
                        ma2 = ma2max
                        icga = ma1min * (j2 + 1) + ma2max
                        for ia in range(na):
                            suma1_r += cgblock[icga] * (
                                        u1_r[ma1] * u2_r[ma2] - u1_i[ma1] *
                                        u2_i[ma2])
                            suma1_i += cgblock[icga] * (
                                        u1_r[ma1] * u2_i[ma2] + u1_i[ma1] *
                                        u2_r[ma2])
                            ma1 += 1
                            ma2 -= 1
                            icga += j2
                        zlist_r[jjz] += cgblock[icgb] * suma1_r
                        zlist_i[jjz] += cgblock[icgb] * suma1_i
                        jju1 += j1 + 1
                        jju2 -= j2 + 1
                        icgb += j2

                    if self.bnorm_flag:
                        zlist_r[jjz] /= (j + 1)
                        zlist_i[jjz] /= (j + 1)
                idouble += 1
        return zlist_r, zlist_i

    def __compute_bi(self, ulisttot_r, ulisttot_i, zlist_r, zlist_i):
        # For now set the number of elements to 1.
        # This also has some implications for the rest of the function.
        # This currently really only works for one element.
        number_elements = 1
        number_element_pairs = number_elements*number_elements
        number_element_triples = number_element_pairs*number_elements
        ielem = 0
        blist = np.zeros(self.idxb_max*number_element_triples)
        itriple = 0
        idouble = 0

        if self.bzero_flag:
            wself = 1.0
            bzero = np.zeros(self.parameters.bispectrum_twojmax+1)
            www = wself * wself * wself
            for j in range(self.parameters.bispectrum_twojmax + 1):
                if self.bnorm_flag:
                    bzero[j] = www
                else:
                    bzero[j] = www * (j + 1)

        for elem1 in range(number_elements):
            for elem2 in range(number_elements):
                for elem3 in range(number_elements):
                    for jjb in range(self.idxb_max):
                        j1 = int(self.idxb[jjb].j1)
                        j2 = int(self.idxb[jjb].j2)
                        j = int(self.idxb[jjb].j)
                        jjz = int(self.idxz_block[j1][j2][j])
                        jju = int(self.idxu_block[j])
                        sumzu = 0.0
                        for mb in range(j // 2):
                            for ma in range(j + 1):
                                sumzu += ulisttot_r[elem3 * self.idxu_max + jju] * \
                                         zlist_r[jjz] + ulisttot_i[
                                             elem3 * self.idxu_max + jju] * zlist_i[
                                             jjz]
                                jjz += 1
                                jju += 1
                        if j % 2 == 0:
                            mb = j // 2
                            for ma in range(mb):
                                sumzu += ulisttot_r[elem3 * self.idxu_max + jju] * \
                                         zlist_r[jjz] + ulisttot_i[
                                             elem3 * self.idxu_max + jju] * zlist_i[
                                             jjz]
                                jjz += 1
                                jju += 1
                            sumzu += 0.5 * (
                                        ulisttot_r[elem3 * self.idxu_max + jju] *
                                        zlist_r[jjz] + ulisttot_i[
                                            elem3 * self.idxu_max + jju] * zlist_i[
                                            jjz])
                        blist[itriple * self.idxb_max + jjb] = 2.0 * sumzu
                    itriple += 1
                idouble += 1

        if self.bzero_flag:
            if not self.wselfall_flag:
                itriple = (ielem * number_elements + ielem) * number_elements + ielem
                for jjb in range(self.idxb_max):
                    j = self.idxb[jjb].j
                    blist[itriple * self.idxb_max + jjb] -= bzero[j]
            else:
                itriple = 0
                for elem1 in range(number_elements):
                    for elem2 in range(number_elements):
                        for elem3 in range(number_elements):
                            for jjb in range(self.idxb_max):
                                j = self.idxb[jjb].j
                                blist[itriple * self.idxb_max + jjb] -= bzero[j]
                            itriple += 1

        return blist
