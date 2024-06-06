"""ACE descriptor class."""

import os
import itertools
import sys

import ase
import ase.io

from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance
from scipy import special

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
        self.ionic_radii = {
            "H": 0.25,
            "He": 1.2,
            "Li": 1.45,
            "Be": 1.05,
            "B": 0.85,
            "C": 0.7,
            "N": 0.65,
            "O": 0.6,
            "F": 0.5,
            "Ne": 1.6,
            "Na": 1.8,
            "Mg": 1.5,
            "Al": 1.25,
            "Si": 1.1,
            "P": 1.0,
            "S": 1.0,
            "Cl": 1.0,
            "Ar": 0.71,
            "K": 2.2,
            "Ca": 1.8,
            "Sc": 1.6,
            "Ti": 1.4,
            "V": 1.35,
            "Cr": 1.4,
            "Mn": 1.4,
            "Fe": 1.4,
            "Co": 1.35,
            "Ni": 1.35,
            "Cu": 1.35,
            "Zn": 1.35,
            "Ga": 1.3,
            "Ge": 1.25,
            "As": 1.15,
            "Se": 1.15,
            "Br": 1.15,
            "Kr": np.nan,
            "Rb": 2.35,
            "Sr": 2.0,
            "Y": 1.8,
            "Zr": 1.55,
            "Nb": 1.45,
            "Mo": 1.45,
            "Tc": 1.35,
            "Ru": 1.3,
            "Rh": 1.35,
            "Pd": 1.4,
            "Ag": 1.6,
            "Cd": 1.55,
            "In": 1.55,
            "Sn": 1.45,
            "Sb": 1.45,
            "Te": 1.4,
            "I": 1.4,
            "Xe": np.nan,
            "Cs": 2.6,
            "Ba": 2.15,
            "La": 1.95,
            "Ce": 1.85,
            "Pr": 1.85,
            "Nd": 1.85,
            "Pm": 1.85,
            "Sm": 1.85,
            "Eu": 1.85,
            "Gd": 1.8,
            "Tb": 1.75,
            "Dy": 1.75,
            "Ho": 1.75,
            "Er": 1.75,
            "Tm": 1.75,
            "Yb": 1.75,
            "Lu": 1.75,
            "Hf": 1.55,
            "Ta": 1.45,
            "W": 1.35,
            "Re": 1.35,
            "Os": 1.3,
            "Ir": 1.35,
            "Pt": 1.35,
            "Au": 1.35,
            "Hg": 1.5,
            "Tl": 1.9,
            "Pb": 1.8,
            "Bi": 1.6,
            "Po": 1.9,
            "At": np.nan,
            "Rn": np.nan,
            "Fr": np.nan,
            "Ra": 2.15,
            "Ac": 1.95,
            "Th": 1.8,
            "Pa": 1.8,
            "U": 1.75,
            "Np": 1.75,
            "Pu": 1.75,
            "Am": 1.75,
            "Cm": np.nan,
            "Bk": np.nan,
            "Cf": np.nan,
            "Es": np.nan,
            "Fm": np.nan,
            "Md": np.nan,
            "No": np.nan,
            "Lr": np.nan,
            "Rf": np.nan,
            "Db": np.nan,
            "Sg": np.nan,
            "Bh": np.nan,
            "Hs": np.nan,
            "Mt": np.nan,
            "Ds": np.nan,
            "Rg": np.nan,
            "Cn": np.nan,
            "Nh": np.nan,
            "Fl": np.nan,
            "Mc": np.nan,
            "Lv": np.nan,
            "Ts": np.nan,
            "Og": np.nan,
            "G": 1.35,
        }

        self.metal_list = [
            "Li",
            "Be",
            "Na",
            "Mg",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Cs",
            "Ba",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Fr",
            "Ha",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tb",
            "Yb",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
        ]

        self.Wigner_3j = self.init_wigner_3j(
            self.parameters.ace_lmax_traditional
        )

        self.Clebsch_Gordan = self.init_clebsch_gordan(
            self.parameters.ace_lmax_traditional
        )

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

        # calculate the coupling coefficients
        coupling_coeffs = self.calculate_coupling_coeffs()
        # save the coupling coefficients
        # saving function will go here

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
        self.fingerprint_length = (
            ncols0 + self._calculate_ace_fingerprint_length()
        )
        printout("Fingerprint length = ", self.fingerprint_length)

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
            printout("Fingerprint length from lammps = ", ncols_local)
            if ncols_local != self.fingerprint_length + 3:
                self.fingerprint_length = ncols_local - 3
                # raise Exception("Inconsistent number of features.")

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
        # TODO: this function is not correct
        total_descriptors = 0
        ranks = self.parameters.ace_ranks
        lmax = self.parameters.ace_lmax
        nmax = self.parameters.ace_nmax
        lmin = self.parameters.ace_lmin

        for rank in range(len(ranks)):
            # Number of radial basis functions for the current rank
            num_radial_functions = nmax[rank]

            # Number of angular basis functions for the current rank
            num_angular_functions = lmax[rank] - lmin[rank] + 1

            # Total number of descriptors for the current rank
            rank_descriptors = num_radial_functions * num_angular_functions

            # Add to total descriptors
            total_descriptors += rank_descriptors

        return total_descriptors

    def calculate_coupling_coeffs(self):
        self.bonds = [
            p
            for p in itertools.product(
                self.parameters.ace_elements, self.parameters.ace_elements
            )
        ]

        (
            rc_range,
            rc_default,
            lmb_default,
            rcin_default,
        ) = self.get_default_settings()

        rcutfac = [float(k) for k in rc_default.split()[2:]]
        lmbda = [float(k) for k in lmb_default.split()[2:]]
        assert len(self.bonds) == len(rcutfac) and len(self.bonds) == len(
            lmbda
        ), "you must have rcutfac and lmbda defined for each bond type"
        printout("global max cutoff (angstrom)", max(rcutfac))
        rcinner = [0.0] * len(self.bonds)
        drcinner = [0.0] * len(self.bonds)

        coeffs = None
        ldict = {
            ranki: li
            for ranki, li in zip(
                self.parameters.ace_ranks, self.parameters.ace_lmax
            )
        }
        rankstrlst = ["%s"] * len(self.parameters.ace_ranks)
        rankstr = "".join(rankstrlst) % tuple(self.parameters.ace_ranks)
        lstrlst = ["%s"] * len(self.parameters.ace_ranks)
        lstr = "".join(lstrlst) % tuple(self.parameters.ace_lmax)

        # load or generate generalized coupling coefficients

        if self.parameters.ace_coupling_type == "wig":
            ccs = self.get_wig_coupling(ldict, L_R=self.parameters.ace_L_R)

        elif self.parameters.ace_coupling_type == "cg":
            ccs = self.get_cg_coupling(ldict, L_R=self.parameters.ace_L_R)

        else:
            sys.exit(
                "Coupling type "
                + str(self.parameters.ace_coupling_type)
                + " not recongised"
            )

    def get_default_settings(self):
        rc_range = {bp: None for bp in self.bonds}
        rin_def = {bp: None for bp in self.bonds}
        rc_def = {bp: None for bp in self.bonds}
        fac_per_elem = {key: 0.5 for key in list(self.ionic_radii.keys())}
        fac_per_elem_sub = {
            "H": 0.61,
            "Be": 0.52,
            "W": 0.45,
            "Cr": 0.5,
            "Fe": 0.5,
            "Si": 0.5,
            "V": 0.5,
            "Au": 0.45,
        }
        fac_per_elem.update(fac_per_elem_sub)

        for bond in self.bonds:
            rcmini, rcmaxi = self.default_rc(bond)
            defaultri = (rcmaxi + rcmini) / np.sum(
                [fac_per_elem[elem] * 2 for elem in bond]
            )
            defaultrcinner = (
                rcmini
            ) * 0.25  # 1/4 of shortest ionic bond length
            rc_range[bond] = [rcmini, rcmaxi]
            rc_def[bond] = defaultri * self.parameters.ace_nshell
            rin_def[bond] = defaultrcinner
        # shift = ( max(rc_def.values()) - min(rc_def.values())  )/2
        shift = np.std(list(rc_def.values()))
        # if apply_shift:
        for bond in self.bonds:
            if rc_def[bond] != max(rc_def.values()):
                if self.parameters.ace_apply_shift:
                    rc_def[bond] = rc_def[bond] + shift  # *nshell
                else:
                    rc_def[bond] = rc_def[bond]  # *nshell
        default_lmbs = [i * 0.05 for i in list(rc_def.values())]
        rc_def_lst = ["%1.3f"] * len(self.bonds)
        rc_def_str = "rcutfac = " + "  ".join(b for b in rc_def_lst) % tuple(
            list(rc_def.values())
        )
        lmb_def_lst = ["%1.3f"] * len(self.bonds)
        lmb_def_str = "lambda = " + "  ".join(b for b in lmb_def_lst) % tuple(
            default_lmbs
        )
        rcin_def_lst = ["%1.3f"] * len(self.bonds)
        rcin_def_str = "rcinner = " + "  ".join(
            b for b in rcin_def_lst
        ) % tuple(list(rin_def.values()))
        return rc_range, rc_def_str, lmb_def_str, rcin_def_str

    def default_rc(self, elms):
        elms = sorted(elms)
        elm1, elm2 = tuple(elms)
        try:
            atnum1 = ase.data.atomic_numbers[elm1]
        except KeyError:
            atnum1 = ase.data.atomic_numbers["Au"]
        covr1 = ase.data.covalent_radii[atnum1]
        vdwr1 = ase.data.vdw_radii[atnum1]
        ionr1 = self.ionic_radii[elm1]
        if np.isnan(vdwr1):
            print(
                "NOTE: using dummy VDW radius of 2* covalent radius for %s"
                % elm1
            )
            vdwr1 = 2 * covr1
        try:
            atnum2 = ase.data.atomic_numbers[elm2]
        except KeyError:
            atnum2 = ase.data.atomic_numbers["Au"]
        covr2 = ase.data.covalent_radii[atnum2]
        vdwr2 = ase.data.vdw_radii[atnum2]
        ionr2 = self.ionic_radii[elm2]
        if np.isnan(vdwr2):
            print(
                "NOTE: using dummy VDW radius of 2* covalent radius for %s"
                % elm2
            )
            vdwr2 = 2 * covr2
        # minrc = min([ionr1,ionr2])
        # maxrc = max([vdwr1,vdwr2])
        minbond = ionr1 + ionr2
        if self.parameters.ace_metal_max:
            if elm1 not in self.metal_list and elm2 not in self.metal_list:
                maxbond = vdwr1 + vdwr2
            elif elm1 in self.metal_list and elm2 not in self.metal_list:
                maxbond = ionr1 + vdwr2
                minbond = (ionr1 + ionr2) * 0.8
            elif elm1 in self.metal_list and elm2 in self.metal_list:
                maxbond = ionr1 + ionr2
                minbond = (ionr1 + ionr2) * 0.8
            else:
                maxbond = ionr1 + ionr2
        else:
            maxbond = vdwr1 + vdwr2
        midbond = (maxbond + minbond) / 2
        # by default, return the ionic bond length * number of bonds for minimum
        returnmin = minbond
        if self.parameters.ace_use_vdw:
            # return vdw bond length if requested
            returnmax = maxbond
        else:
            # return hybrid vdw/ionic bonds by default
            returnmax = midbond
        return round(returnmin, 3), round(returnmax, 3)

    def get_wig_coupling(self, ldict, L_R=0, **kwargs):
        # for now we restrict up to rank 4 trees
        M_Rs = list(range(-L_R, L_R + 1))
        # generic coupling for any L_R - support must be added to call
        ranks = list(ldict.keys())
        coupling = {M_R: {rank: {} for rank in ranks} for M_R in M_Rs}

        for M_R in M_Rs:
            for rank in ranks:
                rnk = rank
                ls_per_rnk = self.generate_l_LR(
                    range(ldict[rank] + 1), rank, L_R, M_R, True
                )
                for lstr in ls_per_rnk:
                    l = [int(k) for k in lstr.split(",")]
                    if rank == 1:
                        decomped = self.rank_1_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 2:
                        decomped = self.rank_2_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 3:
                        decomped = self.rank_3_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 4:
                        decomped = self.rank_4_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank > 4:
                        raise ValueError(
                            "Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 4 have been implemented"
                            % rank
                        )
        return coupling

    def get_cg_coupling(self, ldict, L_R=0, use_permutations=True):
        M_Rs = list(range(-L_R, L_R + 1))
        # generic coupling for any L_R - support must be added to call
        ranks = list(ldict.keys())
        coupling = {M_R: {rank: {} for rank in ranks} for M_R in M_Rs}

        for M_R in M_Rs:
            for rank in ranks:
                rnk = rank
                ls_per_rnk = self.generate_l_LR(
                    range(ldict[rank] + 1),
                    rank,
                    L_R,
                    M_R,
                    use_permutations=use_permutations,
                )
                for lstr in ls_per_rnk:
                    l = [int(k) for k in lstr.split(",")]
                    if rank == 1:
                        decomped = self.rank_1_cg_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 2:
                        decomped = self.rank_2_cg_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 3:
                        decomped = self.rank_3_cg_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank == 4:
                        decomped = self.rank_4_cg_tree(l, L_R, M_R)
                        coupling[M_R][rnk][lstr] = decomped
                    elif rank > 4:
                        raise ValueError(
                            "Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 4 have been implemented"
                            % rank
                        )
        return coupling

    def rank_1_tree(self, l, L_R=0, M_R=0):
        # no nodes for rank 1

        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = [()]
        assert l[0] == L_R, "invalid l=%d for irrep L_R = %d" % (l[0], L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1  = - M_R
                conds = m_ints[0] == -M_R
                if conds:
                    w1 = self.Wigner_3j[
                        "%d,%d,%d,%d,%d,%d" % (l[0], m_ints[0], L_R, M_R, 0, 0)
                    ]
                    phase_power = 0
                    phase = (-1) ** phase_power
                    w = phase * w1

                    decomposed[inter][mstr] = float(w)
        return decomposed

    def rank_2_tree(self, l, L_R=0, M_R=0):
        nodes, remainder = self.tree(l)
        node = nodes[0]
        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = [()]

        assert self.check_triangle(
            l[0], l[1], L_R
        ), "invalid l=(%d,%d) for irrep L_R = %d" % (l[0], l[1], L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1 + m_2 = M1
                conds = (m_ints[0] + m_ints[1]) == M_R
                if conds:
                    w1 = self.Wigner_3j[
                        "%d,%d,%d,%d,%d,%d"
                        % (l[0], m_ints[0], l[1], m_ints[1], L_R, -M_R)
                    ]
                    phase_power = L_R - M_R
                    phase = (-1) ** phase_power
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def rank_3_tree(self, l, L_R=0, M_R=0):
        full_inter_tuples = self.tree_l_inters(l, L_R=L_R, M_R=M_R)
        mstrs = self.get_ms(l, M_R)
        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            L1 = inter[0]
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                for M1 in range(-L1, L1 + 1):
                    # m_1 + m_2 = M1
                    # M1 + m_3 = M_R
                    conds = (m_ints[0] + m_ints[1]) == M1 and (
                        M1 + m_ints[2]
                    ) == M_R
                    if conds:
                        w1 = self.Wigner_3j[
                            "%d,%d,%d,%d,%d,%d"
                            % (l[0], m_ints[0], l[1], m_ints[1], L1, -M1)
                        ]
                        w2 = self.Wigner_3j[
                            "%d,%d,%d,%d,%d,%d"
                            % (L1, M1, l[2], m_ints[2], L_R, -M_R)
                        ]
                        phase_power = (L1) - (M1) + (L_R - M_R)
                        phase = (-1) ** phase_power
                        w = phase * w1 * w2
                        decomposed[inter][mstr] = w
        return decomposed

    def rank_4_tree(self, l, L_R=0, M_R=0):
        nodes, remainder = self.tree(l)
        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = self.tree_l_inters(l, L_R=L_R, M_R=M_R)
        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            L1, L2 = inter
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                for M1 in range(-L1, L1 + 1):
                    for M2 in range(-L2, L2 + 1):
                        # m_1 + m_2 = M1
                        # m_4 + m_3 = M2
                        # M1 + M2 = M_R
                        conds = (
                            (m_ints[0] + m_ints[1]) == M1
                            and (m_ints[2] + m_ints[3]) == M2
                            and (M1 + M2) == M_R
                        )
                        if conds:
                            w1 = self.Wigner_3j[
                                "%d,%d,%d,%d,%d,%d"
                                % (l[0], m_ints[0], l[1], m_ints[1], L1, -M1)
                            ]
                            w2 = self.Wigner_3j[
                                "%d,%d,%d,%d,%d,%d"
                                % (l[2], m_ints[2], l[3], m_ints[3], L2, -M2)
                            ]
                            w3 = self.Wigner_3j[
                                "%d,%d,%d,%d,%d,%d"
                                % (L1, M1, L2, M2, L_R, -M_R)
                            ]
                            phase_power = (L1 + L2) - (M1 + M2) + (L_R - M_R)
                            phase = (-1) ** phase_power
                            w = phase * w1 * w2 * w3

                            decomposed[inter][mstr] = w
        return decomposed

    def rank_1_cg_tree(self, l, L_R=0, M_R=0):
        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = [()]
        assert l[0] == L_R, "invalid l=%d for irrep L_R = %d" % (l[0], L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1  = - M_R
                conds = m_ints[0] == M_R
                if conds:
                    w1 = self.Clebsch_Gordan[
                        "%d,%d,%d,%d,%d,%d" % (l[0], m_ints[0], L_R, M_R, 0, 0)
                    ]
                    phase = 1
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def rank_2_cg_tree(self, l, L_R=0, M_R=0):
        nodes, remainder = self.tree(l)
        node = nodes[0]
        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = [()]

        assert self.check_triangle(
            l[0], l[1], L_R
        ), "invalid l=(%d,%d) for irrep L_R = %d" % (l[0], l[1], L_R)

        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                # m_1 + m_2 = M1
                conds = (m_ints[0] + m_ints[1]) == M_R
                if conds:
                    w1 = self.Clebsch_Gordan[
                        "%d,%d,%d,%d,%d,%d"
                        % (l[0], m_ints[0], l[1], m_ints[1], L_R, M_R)
                    ]
                    phase = 1
                    w = phase * w1

                    decomposed[inter][mstr] = w
        return decomposed

    def rank_3_cg_tree(self, l, L_R=0, M_R=0):
        full_inter_tuples = self.tree_l_inters(l, L_R=L_R, M_R=M_R)
        mstrs = self.get_ms(l, M_R)
        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            L1 = inter[0]
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                for M1 in range(-L1, L1 + 1):
                    # m_1 + m_2 = M1
                    # M1 + m_3 = M_R
                    conds = (m_ints[0] + m_ints[1]) == M1 and (
                        M1 + m_ints[2]
                    ) == M_R
                    if conds:
                        w1 = self.Clebsch_Gordan[
                            "%d,%d,%d,%d,%d,%d"
                            % (l[0], m_ints[0], l[1], m_ints[1], L1, M1)
                        ]
                        w2 = self.Clebsch_Gordan[
                            "%d,%d,%d,%d,%d,%d"
                            % (L1, M1, l[2], m_ints[2], L_R, M_R)
                        ]
                        phase = 1
                        w = phase * w1 * w2
                        decomposed[inter][mstr] = w
        return decomposed

    def rank_4_cg_tree(self, l, L_R=0, M_R=0):
        nodes, remainder = self.tree(l)
        mstrs = self.get_ms(l, M_R)
        full_inter_tuples = self.tree_l_inters(l, L_R=L_R, M_R=M_R)
        decomposed = {
            full_inter_tup: {mstr: 0.0 for mstr in mstrs}
            for full_inter_tup in full_inter_tuples
        }

        for inter in full_inter_tuples:
            L1, L2 = inter
            for mstr in mstrs:
                m_ints = [int(b) for b in mstr.split(",")]
                for M1 in range(-L1, L1 + 1):
                    for M2 in range(-L2, L2 + 1):
                        # m_1 + m_2 = M1
                        # m_4 + m_3 = M2
                        # M1 + M2 = M_R
                        conds = (
                            (m_ints[0] + m_ints[1]) == M1
                            and (m_ints[2] + m_ints[3]) == M2
                            and (M1 + M2) == M_R
                        )
                        if conds:
                            w1 = self.Clebsch_Gordan[
                                "%d,%d,%d,%d,%d,%d"
                                % (l[0], m_ints[0], l[1], m_ints[1], L1, M1)
                            ]
                            w2 = self.Clebsch_Gordan[
                                "%d,%d,%d,%d,%d,%d"
                                % (l[2], m_ints[2], l[3], m_ints[3], L2, M2)
                            ]
                            w3 = self.Clebsch_Gordan[
                                "%d,%d,%d,%d,%d,%d"
                                % (L1, M1, L2, M2, L_R, M_R)
                            ]
                            # phase_power = ( L1 + L2 ) - ( M1 + M2 )  + ( L_R - M_R)
                            # phase = (-1) ** phase_power
                            phase = 1
                            w = phase * w1 * w2 * w3

                            decomposed[inter][mstr] = w
        return decomposed

    def get_ms(self, l, M_R=0):
        # retrieves the set of m_i combinations obeying \sum_i m_i = M_R for an arbitrary l vector
        m_ranges = {ind: range(-l[ind], l[ind] + 1) for ind in range(len(l))}
        m_range_arrays = [list(m_ranges[ind]) for ind in range(len(l))]
        m_combos = list(itertools.product(*m_range_arrays))
        first_m_filter = [i for i in m_combos if np.sum(i) == M_R]
        m_list_replace = ["%d"] * len(l)
        m_str_variable = ",".join(b for b in m_list_replace)
        m_strs = [m_str_variable % fmf for fmf in first_m_filter]
        return m_strs

    def tree(self, l):
        # quick construction of tree leaves
        rank = len(l)
        rngs = list(range(0, rank))
        rngs = iter(rngs)
        count = 0
        tup = []
        while count < int(rank / 2):
            c1 = next(rngs)
            c2 = next(rngs)
            tup.append((c1, c2))
            count += 1
        remainder = None
        if rank % 2 != 0:
            remainder = list(range(rank))[-1]
        return tuple(tup), remainder

    # groups a vector of l quantum numbers pairwise
    def vec_nodes(self, vec, nodes, remainder=None):
        vec_by_tups = [tuple([vec[node[0]], vec[node[1]]]) for node in nodes]
        if remainder != None:
            vec_by_tups = vec_by_tups
        return vec_by_tups

    # assuming a pairwise coupling structure, build the "intermediate" angular momenta
    def tree_l_inters(self, l, L_R=0, M_R=0):
        nodes, remainder = self.tree(l)
        rank = len(l)
        if rank >= 3:
            base_node_inters = {
                node: self.get_intermediates_w(l[node[0]], l[node[1]])
                for node in nodes
            }

        full_inter_tuples = []

        if rank == 1:
            full_inter_tuples.append(())
        elif rank == 2:
            full_inter_tuples.append(())
        elif rank == 3:
            L1s = [i for i in base_node_inters[nodes[0]]]
            for L1 in L1s:
                if self.check_triangle(l[remainder], L1, L_R):
                    full_inter_tuples.append(tuple([L1]))
        elif rank == 4:
            L1L2_prod = [
                i
                for i in itertools.product(
                    base_node_inters[nodes[0]], base_node_inters[nodes[1]]
                )
            ]
            for L1L2 in L1L2_prod:
                L1, L2 = L1L2
                if self.check_triangle(L1, L2, L_R):
                    good_tuple = (L1, L2)
                    full_inter_tuples.append(good_tuple)
        else:
            raise ValueError("rank %d not implemented" % rank)

        return full_inter_tuples

    def generate_l_LR(self, lrng, rank, L_R=0, M_R=0, use_permutations=False):
        if L_R % 2 == 0:
            # symmetric w.r.t. inversion
            inv_parity = True
        if L_R % 2 != 0:
            # odd spherical harmonics are antisymmetric w.r.t. inversion
            inv_parity = False
        lmax = max(lrng)
        ls = []

        llst = ["%d"] * rank
        lstr = ",".join(b for b in llst)

        if rank == 1:
            ls.append("%d" % L_R)

        elif rank > 1:
            all_l_perms = [b for b in itertools.product(lrng, repeat=rank)]
            if use_permutations:
                all_ls = all_l_perms.copy()
            elif not use_permutations:
                # eliminate redundant couplings by only considering lexicographically ordered l_i
                all_ls = [
                    ltup for ltup in all_l_perms if ltup == tuple(sorted(ltup))
                ]
            if rank == 2:
                for ltup in all_ls:
                    if inv_parity:
                        parity_flag = np.sum(ltup + (L_R,)) % 2 == 0
                    elif not inv_parity:
                        parity_flag = np.sum(ltup + (L_R,)) % 2 != 0
                    flag = (
                        self.check_triangle(ltup[0], ltup[1], L_R)
                        and parity_flag
                    )
                    if flag:
                        ls.append(lstr % ltup)
            elif rank == 3:
                nodes, remainder = self.tree(list(all_ls[0]))
                for ltup in all_ls:
                    inters = self.tree_l_inters(list(ltup), L_R=L_R)
                    by_node = self.vec_nodes(ltup, nodes, remainder)
                    for inters_i in inters:
                        li_flags = [
                            self.check_triangle(node[0], node[1], inter)
                            for node, inter in zip(by_node, inters_i)
                        ]
                        inter_flags = [
                            self.check_triangle(
                                inters_i[0], ltup[remainder], L_R
                            )
                        ]
                        flags = li_flags + inter_flags
                        if inv_parity:
                            parity_all = np.sum(ltup) % 2 == 0
                        elif not inv_parity:
                            parity_all = np.sum(ltup) % 2 != 0
                        if all(flags) and parity_all:
                            lsub = lstr % ltup
                            if lsub not in ls:
                                ls.append(lsub)
            elif rank == 4:
                nodes, remainder = self.tree(list(all_ls[0]))
                for ltup in all_ls:
                    inters = self.tree_l_inters(list(ltup), L_R=L_R)
                    by_node = self.vec_nodes(ltup, nodes, remainder)
                    for inters_i in inters:
                        li_flags = [
                            self.check_triangle(node[0], node[1], inter)
                            for node, inter in zip(by_node, inters_i)
                        ]
                        inter_flags = [
                            self.check_triangle(inters_i[0], inters_i[1], L_R)
                        ]
                        flags = li_flags + inter_flags
                        if inv_parity:
                            parity_all = np.sum(ltup) % 2 == 0
                        elif not inv_parity:
                            parity_all = np.sum(ltup) % 2 != 0
                        if all(flags) and parity_all:
                            lsub = lstr % ltup
                            if lsub not in ls:
                                ls.append(lsub)

        return ls

    def check_triangle(self, l1, l2, l3):
        # checks triangle condition between |l1+l2| and l3
        lower_bound = np.abs(l1 - l2)
        upper_bound = np.abs(l1 + l2)
        condition = l3 >= lower_bound and l3 <= upper_bound
        return condition

    def get_intermediates(self, l):
        try:
            l = l.split(",")
            l1 = int(l[0])
            l2 = int(l[1])
        except AttributeError:
            l1 = l[0]
            l2 = l[1]

        tris = [i for i in range(abs(l1 - l2), l1 + l2 + 1)]

        ints = [i for i in tris]
        return ints

    # wrapper
    def get_intermediates_w(self, l1, l2):
        l = [l1, l2]
        return self.get_intermediates(l)

    def wigner_3j(self, j1, m1, j2, m2, j3, m3):
        # uses relation between Clebsch-Gordann coefficients and W-3j symbols to evaluate W-3j
        # VERIFIED - wolframalpha.com
        cg = self.clebsch_gordan(j1, m1, j2, m2, j3, -m3)

        num = np.longdouble((-1) ** (j1 - j2 - m3))
        denom = np.longdouble(((2 * j3) + 1) ** (1 / 2))

        return cg * num / denom  # float(num/denom)

    def init_wigner_3j(self, lmax):
        # returns dictionary of all cg coefficients to be used at a given value of lmax
        cg = {}
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(lmax + 1):
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            for m3 in range(-l3, l3 + 1):
                                key = "%d,%d,%d,%d,%d,%d" % (
                                    l1,
                                    m1,
                                    l2,
                                    m2,
                                    l3,
                                    m3,
                                )
                                cg[key] = self.wigner_3j(
                                    l1, m1, l2, m2, l3, m3
                                )
        return cg

    def init_clebsch_gordan(self, lmax):
        # returns dictionary of all cg coefficients to be used at a given value of lmax
        cg = {}
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(lmax + 1):
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            for m3 in range(-l3, l3 + 1):
                                key = "%d,%d,%d,%d,%d,%d" % (
                                    l1,
                                    m1,
                                    l2,
                                    m2,
                                    l3,
                                    m3,
                                )
                                cg[key] = self.clebsch_gordan(
                                    l1, m1, l2, m2, l3, m3
                                )
        return cg

    def clebsch_gordan(self, j1, m1, j2, m2, j3, m3):
        # Clebsch-gordan coefficient calculator based on eqs. 4-5 of:
        # https://hal.inria.fr/hal-01851097/document

        # VERIFIED: test non-zero indices in Wolfram using format ClebschGordan[{j1,m1},{j2,m2},{j3,m3}]
        # rules:
        rule1 = np.abs(j1 - j2) <= j3
        rule2 = j3 <= j1 + j2
        rule3 = m3 == m1 + m2
        rule4 = np.abs(m3) <= j3

        # rules assumed by input
        # assert np.abs(m1) <= j1, 'm1 must be \in {-j1,j1}'
        # assert np.abs(m2) <= j2, 'm2 must be \in {-j2,j2}'

        if rule1 and rule2 and rule3 and rule4:
            # attempting binomial representation
            N1 = np.longdouble((2 * j3) + 1)
            N2 = (
                np.longdouble(special.factorial(j1 + m1, exact=True))
                * np.longdouble(special.factorial(j1 - m1, exact=True))
                * np.longdouble(special.factorial(j2 + m2, exact=True))
                * np.longdouble(special.factorial(j2 - m2, exact=True))
                * np.longdouble(special.factorial(j3 + m3, exact=True))
                * np.longdouble(special.factorial(j3 - m3, exact=True))
            )

            N3 = (
                np.longdouble(special.factorial(j1 + j2 - j3, exact=True))
                * np.longdouble(special.factorial(j1 - j2 + j3, exact=True))
                * np.longdouble(special.factorial(-j1 + j2 + j3, exact=True))
                * np.longdouble(
                    special.factorial(j1 + j2 + j3 + 1, exact=True)
                )
            )

            # N = np.longdouble((N1*N2))/np.longdouble((N3))
            N = np.longdouble(0.0)
            N += np.divide((N1 * N2), N3)

            G = np.longdouble(0.0)

            # k conditions (see eq.5 of https://hal.inria.fr/hal-01851097/document)
            # k  >= 0
            # k <= j1 - m1
            # k <= j2 + m2

            for k in range(0, min([j1 - m1, j2 + m2]) + 1):
                G1 = np.longdouble((-1) ** k)
                G2 = np.longdouble(special.comb(j1 + j2 - j3, k, exact=True))
                G3 = np.longdouble(
                    special.comb(j1 - j2 + j3, j1 - m1 - k, exact=True)
                )
                G4 = np.longdouble(
                    special.comb(-j1 + j2 + j3, j2 + m2 - k, exact=True)
                )
                G += np.longdouble(G1 * G2 * G3 * G4)
            Nsqrt = np.longdouble(0)
            Nsqrt += np.sqrt(N)
            return Nsqrt * G

        else:
            return 0.0
