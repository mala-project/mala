"""Bispectrum descriptor class."""

import os

import ase
import ase.io

from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance

from mala.common.parallelizer import printout
from mala.descriptors.lammps_utils import extract_compute_np
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
        self.__index_u_block = None
        self.__index_u_max = None
        self.__cglist = None
        self.__index_u_one_initialized = None
        self.__index_u_full = None
        self.__index_u_symmetry_pos = None
        self.__index_u_symmetry_neg = None
        self.__index_u1_full = None
        self.__index_u1_symmetry_pos = None
        self.__index_u1_symmetry_neg = None
        self.__rootpq_full_1 = None
        self.__rootpq_full_2 = None
        self.__index_z_u1r = None
        self.__index_z_u1i = None
        self.__index_z_u2r = None
        self.__index_z_u2i = None
        self.__index_z_icga = None
        self.__index_z_icgb = None
        self.__index_z_jjz = None
        self.__index_z_block = None
        self.__index_b_max = None
        self.__index_b = None

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

        # Create LAMMPS instance.
        lammps_dict = {
            "twojmax": self.parameters.bispectrum_twojmax,
            "rcutfac": self.parameters.bispectrum_cutoff,
        }

        self.lammps_temporary_log = os.path.join(
            outdir,
            "lammps_bgrid_log_" + self.calculation_timestamp + ".tmp",
        )
        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("bispectrum")[0]
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    self.parameters.lammps_compute_file = os.path.join(
                        filepath, "in.bgridlocal.python"
                    )
                else:
                    self.parameters.lammps_compute_file = os.path.join(
                        filepath, "in.bgridlocal_defaultproc.python"
                    )
            else:
                self.parameters.lammps_compute_file = os.path.join(
                    filepath, "in.bgrid.python"
                )

        # Do the LAMMPS calculation and clean up.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (
            (self.parameters.bispectrum_twojmax + 2)
            * (self.parameters.bispectrum_twojmax + 3)
            * (self.parameters.bispectrum_twojmax + 4)
        )
        ncoeff = ncoeff // 24  # integer division
        self.fingerprint_length = ncols0 + ncoeff

        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full bispectrum array at
        # the end of this function.
        # This is not necessarily true for the parallel case.
        if self.parameters._configuration["mpi"]:
            nrows_local = extract_compute_np(
                lmp,
                "bgridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_ROWS,
            )
            ncols_local = extract_compute_np(
                lmp,
                "bgridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_COLS,
            )
            if ncols_local != self.fingerprint_length + 3:
                raise Exception("Inconsistent number of features.")

            snap_descriptors_np = extract_compute_np(
                lmp,
                "bgridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                2,
                array_shape=(nrows_local, ncols_local),
                use_fp64=use_fp64,
            )
            self._clean_calculation(lmp, keep_logs)

            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            return snap_descriptors_np, nrows_local

        else:
            # Extract data from LAMMPS calculation.
            snap_descriptors_np = extract_compute_np(
                lmp,
                "bgrid",
                0,
                2,
                (nz, ny, nx, self.fingerprint_length),
                use_fp64=use_fp64,
            )
            self._clean_calculation(lmp, keep_logs)

            # switch from x-fastest to z-fastest order (swaps 0th and 2nd
            # dimension)
            snap_descriptors_np = snap_descriptors_np.transpose([2, 1, 0, 3])
            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            if self.parameters.descriptors_contain_xyz:
                return snap_descriptors_np, nx * ny * nz
            else:
                return snap_descriptors_np[:, :, :, 3:], nx * ny * nz

    def __calculate_python(self, **kwargs):
        """
        Perform bispectrum calculation using python.

        The code used to this end was adapted from the LAMMPS implementation.
        It serves as a fallback option whereever LAMMPS is not available.
        This may be useful, e.g., to students or people getting started with
        MALA who just want to look around. It is not intended for production
        calculations.
        Compared to the LAMMPS implementation, this implementation has quite a
        few limitations. Namely

            - It is roughly an order of magnitude slower for small systems
              and doesn't scale too great (more information on the optimization
              below)
            - It only works for ONE chemical element
            - It has no MPI or GPU support

        Some options are hardcoded in the same manner the LAMMPS implementation
        hard codes them. Compared to the LAMMPS implementation, some
        essentially never used options are not maintained/optimized.
        """
        printout(
            "Using python for descriptor calculation. "
            "The resulting calculation will be slow for "
            "large systems."
        )

        # The entire bispectrum calculation may be extensively profiled.
        profile_calculation = kwargs.get("profile_calculation", False)
        if profile_calculation:
            import time

            timing_distances = 0
            timing_ui = 0
            timing_zi = 0
            timing_bi = 0
            timing_gridpoints = 0

        # Set up the array holding the bispectrum descriptors.
        ncoeff = (
            (self.parameters.bispectrum_twojmax + 2)
            * (self.parameters.bispectrum_twojmax + 3)
            * (self.parameters.bispectrum_twojmax + 4)
        )
        ncoeff = ncoeff // 24  # integer division
        self.fingerprint_length = ncoeff + 3
        bispectrum_np = np.zeros(
            (
                self.grid_dimensions[0],
                self.grid_dimensions[1],
                self.grid_dimensions[2],
                self.fingerprint_length,
            ),
            dtype=np.float64,
        )

        # Create a list of all potentially relevant atoms.
        all_atoms = self._setup_atom_list()

        # These are technically hyperparameters. We currently simply set them
        # to set values for everything.
        self.rmin0 = 0.0
        self.rfac0 = 0.99363
        self.bzero_flag = False
        self.wselfall_flag = False
        # Currently not supported
        self.bnorm_flag = False
        # Currently not supported
        self.quadraticflag = False
        self.number_elements = 1
        self.wself = 1.0

        # What follows is the python implementation of the
        # bispectrum descriptor calculation.
        #
        # It was developed by first copying the code directly and
        # then optimizing it just enough to be usable. LAMMPS is
        # written in C++, and as such, many for-loops which are
        # optimized by the compiler can be employed. This is
        # drastically inefficient in python, so functions were
        # rewritten to use optimized vector-operations
        # (e.g. via numpy) where possible. This requires the
        # precomputation of quite a few index arrays. Thus,
        # this implementation is memory-intensive, which should
        # not be a problem given the intended use.
        #
        # There is still quite some optimization potential here.
        # I have decided to not optimized this code further just
        # now, since we do not know yet whether the bispectrum
        # descriptors will be used indefinitely, or if, e.g.
        # other types of neural networks will be used.
        # The implementation here is fast enough to be used for
        # tests of small test systems during development,
        # which is the sole purpose. If we eventually decide to
        # stick with bispectrum descriptors and feed-forward
        # neural networks, this code can be further optimized and
        # refined. I will leave some guidance below on what to
        # try/what has already been done, should someone else
        # want to give it a try.
        #
        # Final note: if we want to ship MALA with its own
        # bispectrum descriptor calculation to be used at scale,
        # the best way would potentially be via self-maintained
        # C++-functions.

        ########
        # Initialize index arrays.
        #
        # This function initializes a couple of lists of indices for
        # matrix multiplication/summation. By doing so, nested for-loops
        # can be avoided.
        ########

        if profile_calculation:
            t_begin = time.time()
        self.__init_index_arrays()
        if profile_calculation:
            timing_index_init = time.time() - t_begin

        for x in range(0, self.grid_dimensions[0]):
            for y in range(0, self.grid_dimensions[1]):
                for z in range(0, self.grid_dimensions[2]):
                    # Compute the grid point.
                    if profile_calculation:
                        t_grid = time.time()
                    bispectrum_np[x, y, z, 0:3] = self._grid_to_coord(
                        [x, y, z]
                    )

                    ########
                    # Distance matrix calculation.
                    #
                    # Here, the distances to all atoms within our
                    # targeted cutoff are calculated.
                    ########

                    if profile_calculation:
                        t0 = time.time()
                    distances = np.squeeze(
                        distance.cdist(
                            [bispectrum_np[x, y, z, 0:3]], all_atoms
                        )
                    )
                    distances_cutoff = np.squeeze(
                        np.abs(
                            distances[
                                np.argwhere(
                                    distances
                                    < self.parameters.bispectrum_cutoff
                                )
                            ]
                        )
                    )
                    atoms_cutoff = np.squeeze(
                        all_atoms[
                            np.argwhere(
                                distances < self.parameters.bispectrum_cutoff
                            ),
                            :,
                        ],
                        axis=1,
                    )
                    nr_atoms = np.shape(atoms_cutoff)[0]
                    if profile_calculation:
                        timing_distances += time.time() - t0

                    ########
                    # Compute ui.
                    #
                    # This calculates the expansion coefficients of the
                    # hyperspherical harmonics (usually referred to as ui).
                    ########

                    if profile_calculation:
                        t0 = time.time()
                    ulisttot_r, ulisttot_i = self.__compute_ui(
                        nr_atoms,
                        atoms_cutoff,
                        distances_cutoff,
                        bispectrum_np[x, y, z, 0:3],
                    )
                    if profile_calculation:
                        timing_ui += time.time() - t0

                    ########
                    # Compute zi.
                    #
                    # This calculates the bispectrum components through
                    # triple scalar products/Clebsch-Gordan products.
                    ########

                    if profile_calculation:
                        t0 = time.time()
                    zlist_r, zlist_i = self.__compute_zi(
                        ulisttot_r, ulisttot_i
                    )
                    if profile_calculation:
                        timing_zi += time.time() - t0

                    ########
                    # Compute the bispectrum descriptors itself.
                    #
                    # This essentially just extracts the descriptors from
                    # the expansion coeffcients.
                    ########
                    if profile_calculation:
                        t0 = time.time()
                    bispectrum_np[x, y, z, 3:] = self.__compute_bi(
                        ulisttot_r, ulisttot_i, zlist_r, zlist_i
                    )
                    if profile_calculation:
                        timing_gridpoints += time.time() - t_grid
                        timing_bi += time.time() - t0

        if profile_calculation:
            timing_total = time.time() - t_begin
            print("Python-based bispectrum descriptor calculation timing: ")
            print("Index matrix initialization [s]", timing_index_init)
            print("Overall calculation time [s]", timing_total)
            print(
                "Calculation time per gridpoint [s/gridpoint]",
                timing_gridpoints / np.prod(self.grid_dimensions),
            )
            print("Timing contributions per gridpoint: ")
            print(
                "Distance matrix [s/gridpoint]",
                timing_distances / np.prod(self.grid_dimensions),
            )
            print(
                "Compute ui [s/gridpoint]",
                timing_ui / np.prod(self.grid_dimensions),
            )
            print(
                "Compute zi [s/gridpoint]",
                timing_zi / np.prod(self.grid_dimensions),
            )
            print(
                "Compute bi [s/gridpoint]",
                timing_bi / np.prod(self.grid_dimensions),
            )

        if self.parameters.descriptors_contain_xyz:
            return bispectrum_np, np.prod(self.grid_dimensions)
        else:
            self.fingerprint_length -= 3
            return bispectrum_np[:, :, :, 3:], np.prod(self.grid_dimensions)

    ########
    # Functions and helper classes for calculating the bispectrum descriptors.
    #
    # The ZIndices and BIndices classes are useful stand-ins for structs used
    # in the original C++ code.
    ########

    class _ZIndices:

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

    class _BIndices:

        def __init__(self):
            self.j1 = 0
            self.j2 = 0
            self.j = 0

    def __init_index_arrays(self):
        """
        Initialize index arrays.

        This function initializes a couple of lists of indices for
        matrix multiplication/summation. By doing so, nested for-loops
        can be avoided.

        FURTHER OPTIMIZATION: This function relies on nested for-loops.
        They may be optimized. I have not done so, because it is non-trivial
        in some cases and not really needed. These arrays are the same
        for each grid point, so the overall overhead is rather small.
        """
        # Needed for the Clebsch-Gordan product matrices (below)

        def deltacg(j1, j2, j):
            sfaccg = np.math.factorial((j1 + j2 + j) // 2 + 1)
            return np.sqrt(
                np.math.factorial((j1 + j2 - j) // 2)
                * np.math.factorial((j1 - j2 + j) // 2)
                * np.math.factorial((-j1 + j2 + j) // 2)
                / sfaccg
            )

        ########
        # Indices for compute_ui.
        ########

        # First, the ones also used in LAMMPS.
        idxu_count = 0
        self.__index_u_block = np.zeros(self.parameters.bispectrum_twojmax + 1)
        for j in range(0, self.parameters.bispectrum_twojmax + 1):
            self.__index_u_block[j] = idxu_count
            for mb in range(j + 1):
                for ma in range(j + 1):
                    idxu_count += 1
        self.__index_u_max = idxu_count

        rootpqarray = np.zeros(
            (
                self.parameters.bispectrum_twojmax + 2,
                self.parameters.bispectrum_twojmax + 2,
            )
        )
        for p in range(1, self.parameters.bispectrum_twojmax + 1):
            for q in range(1, self.parameters.bispectrum_twojmax + 1):
                rootpqarray[p, q] = np.sqrt(p / q)

        # These are only for optimization purposes.
        self.__index_u_one_initialized = None
        for j in range(0, self.parameters.bispectrum_twojmax + 1):
            stop = (
                self.__index_u_block[j + 1]
                if j < self.parameters.bispectrum_twojmax
                else self.__index_u_max
            )
            if self.__index_u_one_initialized is None:
                self.__index_u_one_initialized = np.arange(
                    self.__index_u_block[j], stop=stop, step=j + 2
                )
            else:
                self.__index_u_one_initialized = np.concatenate(
                    (
                        self.__index_u_one_initialized,
                        np.arange(
                            self.__index_u_block[j], stop=stop, step=j + 2
                        ),
                    )
                )
        self.__index_u_one_initialized = self.__index_u_one_initialized.astype(
            np.int32
        )
        self.__index_u_full = []
        self.__index_u_symmetry_pos = []
        self.__index_u_symmetry_neg = []
        self.__index_u1_full = []
        self.__index_u1_symmetry_pos = []
        self.__index_u1_symmetry_neg = []
        self.__rootpq_full_1 = []
        self.__rootpq_full_2 = []

        for j in range(1, self.parameters.bispectrum_twojmax + 1):
            jju = int(self.__index_u_block[j])
            jjup = int(self.__index_u_block[j - 1])

            for mb in range(0, j // 2 + 1):
                for ma in range(0, j):
                    self.__rootpq_full_1.append(rootpqarray[j - ma][j - mb])
                    self.__rootpq_full_2.append(rootpqarray[ma + 1][j - mb])
                    self.__index_u_full.append(jju)
                    self.__index_u1_full.append(jjup)
                    jju += 1
                    jjup += 1
                jju += 1

            mbpar = 1
            jju = int(self.__index_u_block[j])
            jjup = int(jju + (j + 1) * (j + 1) - 1)

            for mb in range(0, j // 2 + 1):
                mapar = mbpar
                for ma in range(0, j + 1):
                    if mapar == 1:
                        self.__index_u_symmetry_pos.append(jju)
                        self.__index_u1_symmetry_pos.append(jjup)
                    else:
                        self.__index_u_symmetry_neg.append(jju)
                        self.__index_u1_symmetry_neg.append(jjup)
                    mapar = -mapar
                    jju += 1
                    jjup -= 1
                mbpar = -mbpar

        self.__index_u1_full = np.array(self.__index_u1_full)
        self.__rootpq_full_1 = np.array(self.__rootpq_full_1)
        self.__rootpq_full_2 = np.array(self.__rootpq_full_2)

        ########
        # Indices for compute_zi.
        ########

        # First, the ones also used in LAMMPS.
        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            idxz_count += 1
        idxz_max = idxz_count
        idxz = []
        for z in range(idxz_max):
            idxz.append(self._ZIndices())
        self.__index_z_block = np.zeros(
            (
                self.parameters.bispectrum_twojmax + 1,
                self.parameters.bispectrum_twojmax + 1,
                self.parameters.bispectrum_twojmax + 1,
            )
        )

        idxz_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
                    self.__index_z_block[j1][j2][j] = idxz_count

                    for mb in range(j // 2 + 1):
                        for ma in range(j + 1):
                            idxz[idxz_count].j1 = j1
                            idxz[idxz_count].j2 = j2
                            idxz[idxz_count].j = j
                            idxz[idxz_count].ma1min = max(
                                0, (2 * ma - j - j2 + j1) // 2
                            )
                            idxz[idxz_count].ma2max = (
                                2 * ma
                                - j
                                - (2 * idxz[idxz_count].ma1min - j1)
                                + j2
                            ) // 2
                            idxz[idxz_count].na = (
                                min(j1, (2 * ma - j + j2 + j1) // 2)
                                - idxz[idxz_count].ma1min
                                + 1
                            )
                            idxz[idxz_count].mb1min = max(
                                0, (2 * mb - j - j2 + j1) // 2
                            )
                            idxz[idxz_count].mb2max = (
                                2 * mb
                                - j
                                - (2 * idxz[idxz_count].mb1min - j1)
                                + j2
                            ) // 2
                            idxz[idxz_count].nb = (
                                min(j1, (2 * mb - j + j2 + j1) // 2)
                                - idxz[idxz_count].mb1min
                                + 1
                            )

                            jju = self.__index_u_block[j] + (j + 1) * mb + ma
                            idxz[idxz_count].jju = jju

                            idxz_count += 1

        idxcg_block = np.zeros(
            (
                self.parameters.bispectrum_twojmax + 1,
                self.parameters.bispectrum_twojmax + 1,
                self.parameters.bispectrum_twojmax + 1,
            )
        )
        idxcg_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
                    idxcg_block[j1][j2][j] = idxcg_count
                    for m1 in range(j1 + 1):
                        for m2 in range(j2 + 1):
                            idxcg_count += 1
        self.__cglist = np.zeros(idxcg_count)

        idxcg_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
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
                            for z in range(
                                max(
                                    0,
                                    max(
                                        -(j - j2 + aa2) // 2,
                                        -(j - j1 - bb2) // 2,
                                    ),
                                ),
                                min(
                                    (j1 + j2 - j) // 2,
                                    min((j1 - aa2) // 2, (j2 + bb2) // 2),
                                )
                                + 1,
                            ):
                                ifac = -1 if z % 2 else 1
                                cgsum += ifac / (
                                    np.math.factorial(z)
                                    * np.math.factorial((j1 + j2 - j) // 2 - z)
                                    * np.math.factorial((j1 - aa2) // 2 - z)
                                    * np.math.factorial((j2 + bb2) // 2 - z)
                                    * np.math.factorial(
                                        (j - j2 + aa2) // 2 + z
                                    )
                                    * np.math.factorial(
                                        (j - j1 - bb2) // 2 + z
                                    )
                                )
                            cc2 = 2 * m - j
                            dcg = deltacg(j1, j2, j)
                            sfaccg = np.sqrt(
                                np.math.factorial((j1 + aa2) // 2)
                                * np.math.factorial((j1 - aa2) // 2)
                                * np.math.factorial((j2 + bb2) // 2)
                                * np.math.factorial((j2 - bb2) // 2)
                                * np.math.factorial((j + cc2) // 2)
                                * np.math.factorial((j - cc2) // 2)
                                * (j + 1)
                            )
                            self.__cglist[idxcg_count] = cgsum * dcg * sfaccg
                            idxcg_count += 1

        # These are only for optimization purposes.
        self.__index_z_u1r = []
        self.__index_z_u1i = []
        self.__index_z_u2r = []
        self.__index_z_u2i = []
        self.__index_z_icga = []
        self.__index_z_icgb = []
        self.__index_z_jjz = []
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
            jju1 = int(self.__index_u_block[j1] + (j1 + 1) * mb1min)
            jju2 = int(self.__index_u_block[j2] + (j2 + 1) * mb2max)

            icgb = mb1min * (j2 + 1) + mb2max
            for ib in range(nb):
                ma1 = ma1min
                ma2 = ma2max
                icga = ma1min * (j2 + 1) + ma2max
                for ia in range(na):
                    self.__index_z_jjz.append(jjz)
                    self.__index_z_icgb.append(
                        int(idxcg_block[j1][j2][j]) + icgb
                    )
                    self.__index_z_icga.append(
                        int(idxcg_block[j1][j2][j]) + icga
                    )
                    self.__index_z_u1r.append(jju1 + ma1)
                    self.__index_z_u1i.append(jju1 + ma1)
                    self.__index_z_u2r.append(jju2 + ma2)
                    self.__index_z_u2i.append(jju2 + ma2)
                    ma1 += 1
                    ma2 -= 1
                    icga += j2
                jju1 += j1 + 1
                jju2 -= j2 + 1
                icgb += j2

        self.__index_z_u1r = np.array(self.__index_z_u1r)
        self.__index_z_u1i = np.array(self.__index_z_u1i)
        self.__index_z_u2r = np.array(self.__index_z_u2r)
        self.__index_z_u2i = np.array(self.__index_z_u2i)
        self.__index_z_icga = np.array(self.__index_z_icga)
        self.__index_z_icgb = np.array(self.__index_z_icgb)
        self.__index_z_jjz = np.array(self.__index_z_jjz)

        ########
        # Indices for compute_bi.
        ########

        # These are identical to LAMMPS, because we do not optimize compute_bi.
        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
                    if j >= j1:
                        idxb_count += 1
        self.__index_b_max = idxb_count
        self.__index_b = []
        for b in range(self.__index_b_max):
            self.__index_b.append(self._BIndices())

        idxb_count = 0
        for j1 in range(self.parameters.bispectrum_twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(
                    j1 - j2,
                    min(self.parameters.bispectrum_twojmax, j1 + j2) + 1,
                    2,
                ):
                    if j >= j1:
                        self.__index_b[idxb_count].j1 = j1
                        self.__index_b[idxb_count].j2 = j2
                        self.__index_b[idxb_count].j = j
                        idxb_count += 1

    def __compute_ui(self, nr_atoms, atoms_cutoff, distances_cutoff, grid):
        """
        Compute ui.

        This calculates the expansion coefficients of the
        hyperspherical harmonics (usually referred to as ui).

        FURTHER OPTIMIZATION: This originally was a huge nested for-loop.
        By vectorizing over the atoms and pre-initializing a bunch of arrays,
        a  massive amount of time could be saved. There is one principal
        for-loop remaining - I have not found an easy way to optimize it out.
        Also, I have not tried numba or some other just-in-time compilation,
        may help.
        """
        # Precompute and prepare ui stuff
        theta0 = (
            (distances_cutoff - self.rmin0)
            * self.rfac0
            * np.pi
            / (self.parameters.bispectrum_cutoff - self.rmin0)
        )
        z0 = np.squeeze(distances_cutoff / np.tan(theta0))

        ulist_r_ij = np.zeros((nr_atoms, self.__index_u_max), dtype=np.float64)
        ulist_r_ij[:, 0] = 1.0
        ulist_i_ij = np.zeros((nr_atoms, self.__index_u_max), dtype=np.float64)
        ulisttot_r = np.zeros(self.__index_u_max, dtype=np.float64)
        ulisttot_i = np.zeros(self.__index_u_max, dtype=np.float64)
        r0inv = np.squeeze(
            1.0 / np.sqrt(distances_cutoff * distances_cutoff + z0 * z0)
        )
        ulisttot_r[self.__index_u_one_initialized] = 1.0
        distance_vector = -1.0 * (atoms_cutoff - grid)

        # Cayley-Klein parameters for unit quaternion.
        if nr_atoms > 0:
            a_r = r0inv * z0
            a_i = -r0inv * distance_vector[:, 2]
            b_r = r0inv * distance_vector[:, 1]
            b_i = -r0inv * distance_vector[:, 0]

            # This encapsulates the compute_uarray function
            jju1 = 0
            jju2 = 0
            jju3 = 0
            for jju_outer in range(self.__index_u_max):
                if jju_outer in self.__index_u_full:
                    rootpq = self.__rootpq_full_1[jju1]
                    ulist_r_ij[:, self.__index_u_full[jju1]] += rootpq * (
                        a_r * ulist_r_ij[:, self.__index_u1_full[jju1]]
                        + a_i * ulist_i_ij[:, self.__index_u1_full[jju1]]
                    )
                    ulist_i_ij[:, self.__index_u_full[jju1]] += rootpq * (
                        a_r * ulist_i_ij[:, self.__index_u1_full[jju1]]
                        - a_i * ulist_r_ij[:, self.__index_u1_full[jju1]]
                    )

                    rootpq = self.__rootpq_full_2[jju1]
                    ulist_r_ij[:, self.__index_u_full[jju1] + 1] = (
                        -1.0
                        * rootpq
                        * (
                            b_r * ulist_r_ij[:, self.__index_u1_full[jju1]]
                            + b_i * ulist_i_ij[:, self.__index_u1_full[jju1]]
                        )
                    )
                    ulist_i_ij[:, self.__index_u_full[jju1] + 1] = (
                        -1.0
                        * rootpq
                        * (
                            b_r * ulist_i_ij[:, self.__index_u1_full[jju1]]
                            - b_i * ulist_r_ij[:, self.__index_u1_full[jju1]]
                        )
                    )
                    jju1 += 1
                if jju_outer in self.__index_u1_symmetry_pos:
                    ulist_r_ij[:, self.__index_u1_symmetry_pos[jju2]] = (
                        ulist_r_ij[:, self.__index_u_symmetry_pos[jju2]]
                    )
                    ulist_i_ij[:, self.__index_u1_symmetry_pos[jju2]] = (
                        -ulist_i_ij[:, self.__index_u_symmetry_pos[jju2]]
                    )
                    jju2 += 1

                if jju_outer in self.__index_u1_symmetry_neg:
                    ulist_r_ij[:, self.__index_u1_symmetry_neg[jju3]] = (
                        -ulist_r_ij[:, self.__index_u_symmetry_neg[jju3]]
                    )
                    ulist_i_ij[:, self.__index_u1_symmetry_neg[jju3]] = (
                        ulist_i_ij[:, self.__index_u_symmetry_neg[jju3]]
                    )
                    jju3 += 1

            # This emulates add_uarraytot.
            # First, we compute sfac.
            sfac = np.zeros(nr_atoms)
            if self.parameters.bispectrum_switchflag == 0:
                sfac += 1.0
            else:
                rcutfac = np.pi / (
                    self.parameters.bispectrum_cutoff - self.rmin0
                )
                if nr_atoms > 1:
                    sfac = 0.5 * (
                        np.cos((distances_cutoff - self.rmin0) * rcutfac) + 1.0
                    )
                    sfac[np.where(distances_cutoff <= self.rmin0)] = 1.0
                    sfac[
                        np.where(
                            distances_cutoff
                            > self.parameters.bispectrum_cutoff
                        )
                    ] = 0.0
                else:
                    sfac = 1.0 if distances_cutoff <= self.rmin0 else sfac
                    sfac = 0.0 if distances_cutoff <= self.rmin0 else sfac

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
            for jju in range(self.__index_u_max):
                ulisttot_r[jju] += np.sum(sfac * ulist_r_ij[:, jju])
                ulisttot_i[jju] += np.sum(sfac * ulist_i_ij[:, jju])

        return ulisttot_r, ulisttot_i

    def __compute_zi(self, ulisttot_r, ulisttot_i):
        """
        Compute zi.

        This calculates the bispectrum components through
        triple scalar products/Clebsch-Gordan products.

        FURTHER OPTIMIZATION: In the original code, this is a huge nested
        for-loop. Even after optimization, this is the principal
        computational cost (for realistic systems). I have found this
        implementation to be the most efficient without any major refactoring.
        However, due to the usage of np.unique, numba cannot trivially be used.
        A different route that then may employ just-in-time compilation
        could be fruitful.
        """
        tmp_real = (
            self.__cglist[self.__index_z_icgb]
            * self.__cglist[self.__index_z_icga]
            * (
                ulisttot_r[self.__index_z_u1r] * ulisttot_r[self.__index_z_u2r]
                - ulisttot_i[self.__index_z_u1i]
                * ulisttot_i[self.__index_z_u2i]
            )
        )
        tmp_imag = (
            self.__cglist[self.__index_z_icgb]
            * self.__cglist[self.__index_z_icga]
            * (
                ulisttot_r[self.__index_z_u1r] * ulisttot_i[self.__index_z_u2i]
                + ulisttot_i[self.__index_z_u1i]
                * ulisttot_r[self.__index_z_u2r]
            )
        )

        # Summation over an array based on indices stored in a different
        # array.
        # Taken from: https://stackoverflow.com/questions/67108215/how-to-get-sum-of-values-in-a-numpy-array-based-on-another-array-with-repetitive
        # Under "much better version".
        _, idx, _ = np.unique(
            self.__index_z_jjz, return_counts=True, return_inverse=True
        )
        zlist_r = np.bincount(idx, tmp_real)
        _, idx, _ = np.unique(
            self.__index_z_jjz, return_counts=True, return_inverse=True
        )
        zlist_i = np.bincount(idx, tmp_imag)

        # Commented out for efficiency reasons. May be commented in at a later
        # point if needed.
        # if bnorm_flag:
        #     zlist_r[jjz] /= (j + 1)
        #     zlist_i[jjz] /= (j + 1)
        return zlist_r, zlist_i

    def __compute_bi(self, ulisttot_r, ulisttot_i, zlist_r, zlist_i):
        """
        Compute the bispectrum descriptors itself.

        This essentially just extracts the descriptors from
        the expansion coeffcients.

        FURTHER OPTIMIZATION: I have not optimized this function AT ALL.
        This is due to the fact that its computational footprint is miniscule
        compared to the other parts of the bispectrum descriptor calculation.
        It contains multiple for-loops, that may be optimized out.
        """
        # For now set the number of elements to 1.
        # This also has some implications for the rest of the function.
        # This currently really only works for one element.
        number_elements = 1
        number_element_pairs = number_elements * number_elements
        number_element_triples = number_element_pairs * number_elements
        ielem = 0
        blist = np.zeros(self.__index_b_max * number_element_triples)
        itriple = 0
        idouble = 0

        if self.bzero_flag:
            wself = 1.0
            bzero = np.zeros(self.parameters.bispectrum_twojmax + 1)
            www = wself * wself * wself
            for j in range(self.parameters.bispectrum_twojmax + 1):
                if self.bnorm_flag:
                    bzero[j] = www
                else:
                    bzero[j] = www * (j + 1)

        for elem1 in range(number_elements):
            for elem2 in range(number_elements):
                for elem3 in range(number_elements):
                    for jjb in range(self.__index_b_max):
                        j1 = int(self.__index_b[jjb].j1)
                        j2 = int(self.__index_b[jjb].j2)
                        j = int(self.__index_b[jjb].j)
                        jjz = int(self.__index_z_block[j1][j2][j])
                        jju = int(self.__index_u_block[j])
                        sumzu = 0.0
                        for mb in range(int(np.ceil(j / 2))):
                            for ma in range(j + 1):
                                sumzu += (
                                    ulisttot_r[
                                        elem3 * self.__index_u_max + jju
                                    ]
                                    * zlist_r[jjz]
                                    + ulisttot_i[
                                        elem3 * self.__index_u_max + jju
                                    ]
                                    * zlist_i[jjz]
                                )
                                jjz += 1
                                jju += 1
                        if j % 2 == 0:
                            mb = j // 2
                            for ma in range(mb):
                                sumzu += (
                                    ulisttot_r[
                                        elem3 * self.__index_u_max + jju
                                    ]
                                    * zlist_r[jjz]
                                    + ulisttot_i[
                                        elem3 * self.__index_u_max + jju
                                    ]
                                    * zlist_i[jjz]
                                )
                                jjz += 1
                                jju += 1
                            sumzu += 0.5 * (
                                ulisttot_r[elem3 * self.__index_u_max + jju]
                                * zlist_r[jjz]
                                + ulisttot_i[elem3 * self.__index_u_max + jju]
                                * zlist_i[jjz]
                            )
                        blist[itriple * self.__index_b_max + jjb] = 2.0 * sumzu
                    itriple += 1
                idouble += 1

        if self.bzero_flag:
            if not self.wselfall_flag:
                itriple = (
                    ielem * number_elements + ielem
                ) * number_elements + ielem
                for jjb in range(self.__index_b_max):
                    j = self.__index_b[jjb].j
                    blist[itriple * self.__index_b_max + jjb] -= bzero[j]
            else:
                itriple = 0
                for elem1 in range(number_elements):
                    for elem2 in range(number_elements):
                        for elem3 in range(number_elements):
                            for jjb in range(self.__index_b_max):
                                j = self.__index_b[jjb].j
                                blist[
                                    itriple * self.__index_b_max + jjb
                                ] -= bzero[j]
                            itriple += 1

        # Untested  & Unoptimized
        if self.quadraticflag:
            xyz_length = 3 if self.parameters.descriptors_contain_xyz else 0
            ncount = self.fingerprint_length - xyz_length
            for icoeff in range(ncount):
                bveci = blist[icoeff]
                blist[3 + ncount] = 0.5 * bveci * bveci
                ncount += 1
                for jcoeff in range(icoeff + 1, ncount):
                    blist[xyz_length + ncount] = bveci * blist[jcoeff]
                    ncount += 1

        return blist
