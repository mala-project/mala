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

        # Index arrays needed only when computing the bispectrum descriptors
        # via python.
        # They are later filled in the __init_index_arrays() function.
        self.__idxu_block = None
        self.__idxu_max = None
        self.__cglist = None
        self.__idxu_init_pairs = None
        self.__all_jju = None
        self.__all_pos_jju = None
        self.__all_neg_jju = None
        self.__all_jjup = None
        self.__all_pos_jjup = None
        self.__all_neg_jjup = None
        self.__all_rootpq_1 = None
        self.__all_rootpq_2 = None
        self.__zsum_u1r = None
        self.__zsum_u1i = None
        self.__zsum_u2r = None
        self.__zsum_u2i = None
        self.__zsum_icga = None
        self.__zsum_icgb = None
        self.__zsum_jjz = None
        self.__idxz_block = None
        self.__idxb_max = None
        self.__idxb = None

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
        lammps_dict = {"twojmax": self.parameters.bispectrum_twojmax,
                       "rcutfac": self.parameters.bispectrum_cutoff,
                       "atom_config_fname": ase_out_path}
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
                 (self.parameters.bispectrum_twojmax + 3) * \
                 (self.parameters.bispectrum_twojmax + 4)
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
        import time
        ncoeff = (self.parameters.bispectrum_twojmax + 2) * \
                 (self.parameters.bispectrum_twojmax + 3) * \
                 (self.parameters.bispectrum_twojmax + 4)
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
        # Currently not working if True
        self.bnorm_flag = False
        self.quadraticflag = False
        self.number_elements = 1
        self.wself = 1.0

        t0 = time.time()
        self.__init_index_arrays()
        # print("Init index arrays", time.time()-t0)
        for x in range(0, self.grid_dimensions[0]):
            for y in range(0, self.grid_dimensions[1]):
                for z in range(0, self.grid_dimensions[2]):
                    # Compute the grid.
                    bispectrum_np[x, y, z, 0:3] = \
                        self._grid_to_coord([x, y, z])

                    # Compute the bispectrum descriptors.
                    t0 = time.time()
                    t00 = time.time()
                    distances = np.squeeze(distance.cdist(
                        [bispectrum_np[x, y, z, 0:3]],
                        all_atoms))
                    distances_cutoff = np.squeeze(np.abs(
                        distances[np.argwhere(
                            distances < self.parameters.bispectrum_cutoff)]))
                    atoms_cutoff = np.squeeze(
                        all_atoms[np.argwhere(
                            distances < self.parameters.bispectrum_cutoff), :])
                    nr_atoms = np.shape(atoms_cutoff)[0]
                    # print("Distances", time.time() - t0)

                    printer = False
                    if x == 0 and y == 0 and z == 1:
                        printer = True

                    t0 = time.time()
                    ulisttot_r, ulisttot_i = \
                        self.__compute_ui(nr_atoms, atoms_cutoff,
                                          distances_cutoff,
                                          bispectrum_np[x, y, z, 0:3])
                    # print("Compute ui", time.time() - t0)

                    t0 = time.time()
                    # zlist_r, zlist_i = \
                    #     self.__compute_zi(ulisttot_r, ulisttot_i, printer)
                    zlist_r, zlist_i = \
                        self.__compute_zi(ulisttot_r, ulisttot_i)
                    # print("Compute zi", time.time() - t0)

                    t0 = time.time()
                    bispectrum_np[x, y, z, 3:] = \
                        self.__compute_bi(ulisttot_r, ulisttot_i, zlist_r,
                                          zlist_i, printer)
                    # print("Compute bi", time.time() - t0)
                    # print("Per grid point", time.time()-t00)

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
        def deltacg(j1, j2, j):
            sfaccg = np.math.factorial((j1 + j2 + j) // 2 + 1)
            return np.sqrt(np.math.factorial((j1 + j2 - j) // 2) *
                           np.math.factorial((j1 - j2 + j) // 2) *
                           np.math.factorial((-j1 + j2 + j) // 2) / sfaccg)

        idxu_count = 0
        self.__idxu_block = np.zeros(self.parameters.bispectrum_twojmax + 1)
        for j in range(0, self.parameters.bispectrum_twojmax + 1):
            self.__idxu_block[j] = idxu_count
            for mb in range(j + 1):
                for ma in range(j + 1):
                    idxu_count += 1
        self.__idxu_max = idxu_count

        rootpqarray = np.zeros((self.parameters.bispectrum_twojmax + 2,
                                self.parameters.bispectrum_twojmax + 2))
        for p in range(1, self.parameters.bispectrum_twojmax + 1):
            for q in range(1,
                           self.parameters.bispectrum_twojmax + 1):
                rootpqarray[p, q] = np.sqrt(p / q)

        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            idxz_count += 1
        idxz_max = idxz_count
        idxz = []
        for z in range(idxz_max):
            idxz.append(self.ZIndices())
        self.__idxz_block = np.zeros((self.parameters.bispectrum_twojmax + 1,
                                      self.parameters.bispectrum_twojmax + 1,
                                      self.parameters.bispectrum_twojmax + 1))

        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    self.__idxz_block[j1][j2][j] = idxz_count

                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            idxz[idxz_count].j1 = j1
                            idxz[idxz_count].j2 = j2
                            idxz[idxz_count].j = j
                            idxz[idxz_count].ma1min = max(0, (
                                        2 * ma - j - j2 + j1) // 2)
                            idxz[idxz_count].ma2max = (2 * ma - j - (2 * idxz[
                                idxz_count].ma1min - j1) + j2) // 2
                            idxz[idxz_count].na = min(j1, (
                                        2 * ma - j + j2 + j1) // 2) - idxz[
                                                      idxz_count].ma1min + 1
                            idxz[idxz_count].mb1min = max(0, (
                                        2 * mb - j - j2 + j1) // 2)
                            idxz[idxz_count].mb2max = (2 * mb - j - (2 * idxz[
                                idxz_count].mb1min - j1) + j2) // 2
                            idxz[idxz_count].nb = min(j1, (
                                        2 * mb - j + j2 + j1) // 2) - idxz[
                                                      idxz_count].mb1min + 1

                            jju = self.__idxu_block[j] + (j + 1) * mb + ma
                            idxz[idxz_count].jju = jju

                            idxz_count += 1

        idxcg_block = np.zeros((self.parameters.bispectrum_twojmax + 1,
                                     self.parameters.bispectrum_twojmax + 1,
                                     self.parameters.bispectrum_twojmax + 1))
        idxcg_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    idxcg_block[j1][j2][j] = idxcg_count
                    for m1 in range(j1 + 1):
                        for m2 in range(j2 + 1):
                            idxcg_count += 1
        self.__cglist = np.zeros(idxcg_count)

        idxcg_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    for m1 in range(j1 + 1):
                        aa2 = 2 * m1 - j1
                        for m2 in range(j2 + 1):
                            bb2 = 2 * m2 - j2
                            m = (aa2 + bb2 + j) // 2
                            if m < 0 or m > j:
                                self.__cglist[idxcg_count] = 0.0
                                idxcg_count += 1
                                continue
                            cgsum = 0.0
                            for z in range(max(0, max(-(j - j2 + aa2) // 2,
                                                      -(j - j1 - bb2) // 2)),
                                           min((j1 + j2 - j) // 2,
                                               min((j1 - aa2) // 2,
                                                   (j2 + bb2) // 2)) + 1):
                                ifac = -1 if z % 2 else 1
                                cgsum += ifac / (np.math.factorial(z) * np.math.factorial(
                                    (j1 + j2 - j) // 2 - z) * np.math.factorial(
                                    (j1 - aa2) // 2 - z) * np.math.factorial(
                                    (j2 + bb2) // 2 - z) * np.math.factorial(
                                    (j - j2 + aa2) // 2 + z) * np.math.factorial(
                                    (j - j1 - bb2) // 2 + z))
                            cc2 = 2 * m - j
                            dcg = deltacg(j1, j2, j)
                            sfaccg = np.sqrt(
                                np.math.factorial((j1 + aa2) // 2) * np.math.factorial(
                                    (j1 - aa2) // 2) * np.math.factorial(
                                    (j2 + bb2) // 2) * np.math.factorial(
                                    (j2 - bb2) // 2) * np.math.factorial(
                                    (j + cc2) // 2) * np.math.factorial(
                                    (j - cc2) // 2) * (j + 1))
                            self.__cglist[idxcg_count] = cgsum * dcg * sfaccg
                            idxcg_count += 1

        # BEGINNING OF UI/ZI OPTIMIZATION BLOCK!
        # Everthing in this block is EXCLUSIVELY for the
        # optimization of compute_ui and compute_zi!
        # Declaring indices over which to perform vector operations speeds
        # things up significantly - it is not memory-sparse, but this is
        # not a big concern for the python implementation which is only
        # used for small systems anyway.
        self.__idxu_init_pairs = None
        for j in range(0, self.parameters.bispectrum_twojmax + 1):
            stop = self.__idxu_block[j + 1] if j < self.parameters.bispectrum_twojmax else self.__idxu_max
            if self.__idxu_init_pairs is None:
                self.__idxu_init_pairs = np.arange(self.__idxu_block[j], stop=stop, step=j + 2)
            else:
                self.__idxu_init_pairs = np.concatenate((self.__idxu_init_pairs,
                                                         np.arange(self.__idxu_block[j], stop=stop, step=j + 2)))
        self.__idxu_init_pairs = self.__idxu_init_pairs.astype(np.int32)
        self.__all_jju = []
        self.__all_pos_jju = []
        self.__all_neg_jju = []
        self.__all_jjup = []
        self.__all_pos_jjup = []
        self.__all_neg_jjup = []
        self.__all_rootpq_1 = []
        self.__all_rootpq_2 = []

        for j in range(1, self.parameters.bispectrum_twojmax + 1):
            jju = int(self.__idxu_block[j])
            jjup = int(self.__idxu_block[j - 1])

            for mb in range(0, j // 2 + 1):
                for ma in range(0, j):
                    self.__all_rootpq_1.append(rootpqarray[j - ma][j - mb])
                    self.__all_rootpq_2.append(rootpqarray[ma + 1][j - mb])
                    self.__all_jju.append(jju)
                    self.__all_jjup.append(jjup)
                    jju += 1
                    jjup += 1
                jju += 1

            mbpar = 1
            jju = int(self.__idxu_block[j])
            jjup = int(jju + (j + 1) * (j + 1) - 1)

            for mb in range(0, j // 2 + 1):
                mapar = mbpar
                for ma in range(0, j + 1):
                    if mapar == 1:
                        self.__all_pos_jju.append(jju)
                        self.__all_pos_jjup.append(jjup)
                    else:
                        self.__all_neg_jju.append(jju)
                        self.__all_neg_jjup.append(jjup)
                    mapar = -mapar
                    jju += 1
                    jjup -= 1
                mbpar = -mbpar

        self.__all_jjup = np.array(self.__all_jjup)
        self.__all_rootpq_1 = np.array(self.__all_rootpq_1)
        self.__all_rootpq_2 = np.array(self.__all_rootpq_2)

        self.__zsum_u1r = []
        self.__zsum_u1i = []
        self.__zsum_u2r = []
        self.__zsum_u2i = []
        self.__zsum_icga = []
        self.__zsum_icgb = []
        self.__zsum_jjz = []
        for jjz in range(idxz_max):
            j1 = idxz[jjz].j1
            j2 = idxz[jjz].j2
            j = idxz[jjz].j
            ma1min = idxz[jjz].ma1min
            ma2max = idxz[jjz].ma2max
            na = idxz[jjz].na
            mb1min = idxz[jjz].mb1min
            mb2max = idxz[jjz].mb2max
            nb = idxz[jjz].nb
            jju1 = int(self.__idxu_block[j1] + (j1 + 1) * mb1min)
            jju2 = int(self.__idxu_block[j2] + (j2 + 1) * mb2max)

            icgb = mb1min * (j2 + 1) + mb2max
            for ib in range(nb):
                ma1 = ma1min
                ma2 = ma2max
                icga = ma1min * (j2 + 1) + ma2max
                for ia in range(na):
                    self.__zsum_jjz.append(jjz)
                    self.__zsum_icgb.append(int(idxcg_block[j1][j2][j]) + icgb)
                    self.__zsum_icga.append(int(idxcg_block[j1][j2][j]) + icga)
                    self.__zsum_u1r.append(jju1 + ma1)
                    self.__zsum_u1i.append(jju1 + ma1)
                    self.__zsum_u2r.append(jju2 + ma2)
                    self.__zsum_u2i.append(jju2 + ma2)
                    ma1 += 1
                    ma2 -= 1
                    icga += j2
                jju1 += j1 + 1
                jju2 -= j2 + 1
                icgb += j2

        self.__zsum_u1r = np.array(self.__zsum_u1r)
        self.__zsum_u1i = np.array(self.__zsum_u1i)
        self.__zsum_u2r = np.array(self.__zsum_u2r)
        self.__zsum_u2i = np.array(self.__zsum_u2i)
        self.__zsum_icga = np.array(self.__zsum_icga)
        self.__zsum_icgb = np.array(self.__zsum_icgb)
        self.__zsum_jjz = np.array(self.__zsum_jjz)

        # END OF UI/ZI OPTIMIZATION BLOCK!


        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax,
                                            j1 + j2) + 1, 2):
                    if j >= j1:
                        idxb_count += 1
        self.__idxb_max = idxb_count
        self.__idxb = []
        for b in range(self.__idxb_max):
            self.__idxb.append(self.BIndices())

        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(j1 - j2, min(self.parameters.bispectrum_twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        self.__idxb[idxb_count].j1 = j1
                        self.__idxb[idxb_count].j2 = j2
                        self.__idxb[idxb_count].j = j
                        idxb_count += 1

    def __compute_ui(self, nr_atoms, atoms_cutoff, distances_cutoff, grid):
        # Precompute and prepare ui stuff
        theta0 = (distances_cutoff - self.rmin0) * self.rfac0 * np.pi / (
                    self.parameters.bispectrum_cutoff - self.rmin0)
        z0 = np.squeeze(distances_cutoff / np.tan(theta0))

        ulist_r_ij = np.zeros((nr_atoms, self.__idxu_max), dtype=np.float64)
        ulist_r_ij[:, 0] = 1.0
        ulist_i_ij = np.zeros((nr_atoms, self.__idxu_max), dtype=np.float64)
        ulisttot_r = np.zeros(self.__idxu_max, dtype=np.float64)
        ulisttot_i = np.zeros(self.__idxu_max, dtype=np.float64)
        r0inv = np.squeeze(1.0 / np.sqrt(distances_cutoff*distances_cutoff + z0*z0))
        ulisttot_r[self.__idxu_init_pairs] = 1.0
        distance_vector = -1.0 * (atoms_cutoff - grid)
        # Cayley-Klein parameters for unit quaternion.
        a_r = r0inv * z0
        a_i = -r0inv * distance_vector[:,2]
        b_r = r0inv * distance_vector[:,1]
        b_i = -r0inv * distance_vector[:,0]

        # This encapsulates the compute_uarray function
        jju1 = 0
        jju2 = 0
        jju3 = 0
        for jju_outer in range(self.__idxu_max):
            if jju_outer in self.__all_jju:
                rootpq = self.__all_rootpq_1[jju1]
                ulist_r_ij[:, self.__all_jju[jju1]] += rootpq * (
                        a_r * ulist_r_ij[:, self.__all_jjup[jju1]] +
                        a_i *
                        ulist_i_ij[:, self.__all_jjup[jju1]])
                ulist_i_ij[:, self.__all_jju[jju1]] += rootpq * (
                        a_r * ulist_i_ij[:, self.__all_jjup[jju1]] -
                        a_i *
                        ulist_r_ij[:, self.__all_jjup[jju1]])

                rootpq = self.__all_rootpq_2[jju1]
                ulist_r_ij[:, self.__all_jju[jju1] + 1] = -1.0 * rootpq * (
                        b_r * ulist_r_ij[:, self.__all_jjup[jju1]] +
                        b_i *
                        ulist_i_ij[:, self.__all_jjup[jju1]])
                ulist_i_ij[:, self.__all_jju[jju1] + 1] = -1.0 * rootpq * (
                        b_r * ulist_i_ij[:, self.__all_jjup[jju1]] -
                        b_i *
                        ulist_r_ij[:, self.__all_jjup[jju1]])
                jju1 += 1
            if jju_outer in self.__all_pos_jjup:
                ulist_r_ij[:, self.__all_pos_jjup[jju2]] = ulist_r_ij[:,
                                                        self.__all_pos_jju[jju2]]
                ulist_i_ij[:, self.__all_pos_jjup[jju2]] = -ulist_i_ij[:,
                                                         self.__all_pos_jju[jju2]]
                jju2 += 1

            if jju_outer in self.__all_neg_jjup:
                ulist_r_ij[:, self.__all_neg_jjup[jju3]] = -ulist_r_ij[:,
                                                         self.__all_neg_jju[jju3]]
                ulist_i_ij[:, self.__all_neg_jjup[jju3]] = ulist_i_ij[:,
                                                        self.__all_neg_jju[jju3]]
                jju3 += 1

        # This emulates add_uarraytot.
        # First, we compute sfac.
        sfac = np.zeros(nr_atoms)
        if self.parameters.bispectrum_switchflag == 0:
            sfac += 1.0
        else:
            rcutfac = np.pi / (self.parameters.bispectrum_cutoff -
                               self.rmin0)
            sfac = 0.5 * (np.cos((distances_cutoff - self.rmin0) * rcutfac)
                          + 1.0)
            sfac[np.where(distances_cutoff <= self.rmin0)] = 1.0
            sfac[np.where(distances_cutoff >
                          self.parameters.bispectrum_cutoff)] = 0.0

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
        for jju in range(self.__idxu_max):
            ulisttot_r[jju] += np.sum(sfac * ulist_r_ij[:, jju])
            ulisttot_i[jju] += np.sum(sfac * ulist_i_ij[:, jju])

        return ulisttot_r, ulisttot_i

    def __compute_zi(self, ulisttot_r, ulisttot_i):
        tmp_real = self.__cglist[self.__zsum_icgb] * \
                   self.__cglist[self.__zsum_icga] * \
                   (ulisttot_r[self.__zsum_u1r] * ulisttot_r[self.__zsum_u2r]
                    - ulisttot_i[self.__zsum_u1i] * ulisttot_i[self.__zsum_u2i])
        tmp_imag = self.__cglist[self.__zsum_icgb] * \
                   self.__cglist[self.__zsum_icga] * \
                   (ulisttot_r[self.__zsum_u1r] * ulisttot_i[self.__zsum_u2i]
                    + ulisttot_i[self.__zsum_u1i] * ulisttot_r[self.__zsum_u2r])

        # Summation over an array based on indices stored in a different
        # array.
        # Taken from: https://stackoverflow.com/questions/67108215/how-to-get-sum-of-values-in-a-numpy-array-based-on-another-array-with-repetitive
        # Under "much better version".
        _, idx, _ = np.unique(self.__zsum_jjz, return_counts=True,
                              return_inverse=True)
        zlist_r = np.bincount(idx, tmp_real)
        _, idx, _ = np.unique(self.__zsum_jjz, return_counts=True,
                              return_inverse=True)
        zlist_i = np.bincount(idx, tmp_imag)

        # Commented out for efficiency reasons. May be commented in at a later
        # point if needed.
        # if bnorm_flag:
        #     zlist_r[jjz] /= (j + 1)
        #     zlist_i[jjz] /= (j + 1)
        return zlist_r, zlist_i

    def __compute_bi(self, ulisttot_r, ulisttot_i, zlist_r, zlist_i, printer):
        # For now set the number of elements to 1.
        # This also has some implications for the rest of the function.
        # This currently really only works for one element.
        number_elements = 1
        number_element_pairs = number_elements*number_elements
        number_element_triples = number_element_pairs*number_elements
        ielem = 0
        blist = np.zeros(self.__idxb_max * number_element_triples)
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
                    for jjb in range(self.__idxb_max):
                        j1 = int(self.__idxb[jjb].j1)
                        j2 = int(self.__idxb[jjb].j2)
                        j = int(self.__idxb[jjb].j)
                        jjz = int(self.__idxz_block[j1][j2][j])
                        jju = int(self.__idxu_block[j])
                        sumzu = 0.0
                        for mb in range(int(np.ceil(j/2))):
                            for ma in range(j + 1):
                                sumzu += ulisttot_r[elem3 * self.__idxu_max + jju] * \
                                         zlist_r[jjz] + ulisttot_i[
                                             elem3 * self.__idxu_max + jju] * zlist_i[
                                             jjz]
                                jjz += 1
                                jju += 1
                        if j % 2 == 0:
                            mb = j // 2
                            for ma in range(mb):
                                sumzu += ulisttot_r[elem3 * self.__idxu_max + jju] * \
                                         zlist_r[jjz] + ulisttot_i[
                                             elem3 * self.__idxu_max + jju] * zlist_i[
                                             jjz]
                                jjz += 1
                                jju += 1
                            sumzu += 0.5 * (
                                    ulisttot_r[elem3 * self.__idxu_max + jju] *
                                    zlist_r[jjz] + ulisttot_i[
                                        elem3 * self.__idxu_max + jju] * zlist_i[
                                            jjz])
                        blist[itriple * self.__idxb_max + jjb] = 2.0 * sumzu
                    itriple += 1
                idouble += 1

        if self.bzero_flag:
            if not self.wselfall_flag:
                itriple = (ielem * number_elements + ielem) * number_elements + ielem
                for jjb in range(self.__idxb_max):
                    j = self.__idxb[jjb].j
                    blist[itriple * self.__idxb_max + jjb] -= bzero[j]
            else:
                itriple = 0
                for elem1 in range(number_elements):
                    for elem2 in range(number_elements):
                        for elem3 in range(number_elements):
                            for jjb in range(self.__idxb_max):
                                j = self.__idxb[jjb].j
                                blist[itriple * self.__idxb_max + jjb] -= bzero[j]
                            itriple += 1

        # Untested  & Unoptimized
        if self.quadraticflag:
            xyz_length = 3 if self.parameters.descriptors_contain_xyz \
                else 0
            ncount = self.fingerprint_length - xyz_length
            for icoeff in range(ncount):
                bveci = blist[icoeff]
                blist[3 + ncount] = 0.5 * bveci * \
                                                     bveci
                ncount += 1
                for jcoeff in range(icoeff + 1, ncount):
                    blist[xyz_length + ncount] = bveci * \
                                                         blist[
                                                             jcoeff]
                    ncount += 1

        return blist
