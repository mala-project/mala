"""ACE descriptor class."""

import os
import itertools
import sys
import pickle

import ase
import ase.io
from importlib.util import find_spec
import numpy as np
from scipy import special
from mendeleev.fetch import fetch_table

from mala.common.parallelizer import printout
from mala.descriptors.lammps_utils import extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.descriptors.ace_potential import AcePot
import mala.descriptors.ace_coupling_utils as acu
import mala.descriptors.wigner_coupling as wigner_coupling
import mala.descriptors.cg_coupling as cg_coupling


class ACE(Descriptor):
    """Class for calculation and parsing of bispectrum descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(ACE, self).__init__(parameters)

        # Initialize a dictionary with ionic radii (in Angstrom).
        # and a list containing all elements which are considered metals.
        # Both is done via mendeleev. The definition of metal here is
        # "everything in groups 12 and below, including Lanthanoids and
        # Actininoids, excluding hydrogen".
        self.ionic_radii, self.metal_list = self.__init_element_lists()

        # Initialize several arrays that are computed using
        # nested for-loops but can be reused.
        # Within these routines, these arrays are saved to pkl files
        # in the directory where ace.py is located.
        # If a pkl file is detected, the array is loaded from the pkl file,
        # otherwise it is computed and saved to a pkl file.
        # The arrys are used within the ACE descriptor calculation.
        self.__precomputed_wigner_3j = self.__init_wigner_3j(
            self.parameters.ace_lmax_traditional
        )

        self.__precomputed_clebsch_gordan = self.__init_clebsch_gordan(
            self.parameters.ace_lmax_traditional
        )

        # File which holds the coupling coefficients.
        # Can be provided by users, in which case consistency will be checked.
        # If no file is detected, a new file is computed.
        self.couplings_yace_file = None

        # TODO: I am not sure what this does precisely and think we should
        # clarify it.
        print(self.parameters.ace_elements)
        assert (
            not self.parameters.ace_types_like_snap
        ), "Using the same 'element' type for atoms and grid points is not permitted for standar d mala models"
        if (
            self.parameters.ace_types_like_snap
            and "G" in self.parameters.ace_elements
        ):
            raise Exception(
                "for types_like_snap = True, you must remove the separate element type for grid points"
            )

        # TODO: I am not sure what this does precisely and think we should
        # clarify it. Also, is this element list just for reference and thus
        # only contains the grid and Al or is it supposed to contain elements
        # we are calculating?
        if len(self.parameters.ace_elements) != len(
            self.parameters.ace_reference_ens
        ):
            raise Exception(
                "Reference ensemble must match length of element list!"
            )

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "ACE"

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
        Perform ACE descriptor calculation using LAMMPS.

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
        self.setup_lammps_tmp_files("acegrid", outdir)

        ase.io.write(
            self._lammps_temporary_input, self._atoms, format=lammps_format
        )

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Calculate and store the coupling coefficients, if necessary.
        # If coupling coefficients are provided by the user, these are
        # first checked for consistency.
        self.check_coupling_coeffs(self.couplings_yace_file)
        if self.couplings_yace_file is None:
            self.couplings_yace_file = self.calculate_coupling_coeffs()

        # save the coupling coefficients
        # saving function will go here

        # Create LAMMPS instance.
        # TODO: Either rename the cutoff radius or introduce a second
        #  parameter (I would advocate for the latter)
        if self.parameters.bispectrum_cutoff > self.maxrc:
            lammps_dict = {
                "ace_coeff_file": "coupling_coefficients.yace",
                "rcutfac": self.parameters.bispectrum_cutoff,
            }
        else:
            printout(
                "one or more automatically generated ACE rcutfacs is larger than the input rcutfac- updating rcutfac accordingly"
            )
            lammps_dict = {
                "ace_coeff_file": "coupling_coefficients.yace",
                "rcutfac": self.maxrc,
            }

        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        # TODO: Can we get rid of the extra "ace_like_snap" files here?
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("ace")[0]
            if not self.parameters.ace_types_like_snap:
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
            if self.parameters.ace_types_like_snap:
                if self.parameters._configuration["mpi"]:
                    if self.parameters.use_z_splitting:
                        self.parameters.lammps_compute_file = os.path.join(
                            filepath, "in.acegridlocal_s.python"
                        )
                    else:
                        self.parameters.lammps_compute_file = os.path.join(
                            filepath, "in.acegridlocal_defaultproc_s.python"
                        )
                else:
                    self.parameters.lammps_compute_file = os.path.join(
                        filepath, "in.acegrid_s.python"
                    )
        # Do the LAMMPS calculation and clean up.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full bispectrum array at
        # the end of this function.
        # This is not necessarily true for the parallel case.

        # TODO: This is more a LAMMPS thing, but I think it would be nicer
        # if mygridlocal and mygrid would be called acegrid or something,
        # agrid is fine as well, in the bispectrum case we call it bgrid.
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
            if ncols_local != self.feature_size + 3:
                self.feature_size = ncols_local - 3
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

    def check_coupling_coeffs(self, coupling_file):
        if coupling_file is not None:
            with open(coupling_file, "r") as readout:
                lines = readout.readlines()
                element_list = (
                    lines[0]
                    .split(":")[1]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                )
                element_list = [e.strip() for e in element_list]
                print(set(self._atoms.symbols), set(element_list[:-1]))
                # for line in readout:
                #     if "elements" in line:
                #         print(line)

    def calculate_coupling_coeffs(self):
        ncols0 = 3
        # TODO: IIRC, then these coefficients have to be calculated only ONCE
        # per element; so we should maybe think about a way of caching them.

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
        self.maxrc = np.max(
            rcutfac
        )  # set radial cutoff based on automatically generated ACE cutoffs
        lmbda = [float(k) for k in lmb_default.split()[2:]]
        assert len(self.bonds) == len(rcutfac) and len(self.bonds) == len(
            lmbda
        ), "you must have rcutfac and lmbda defined for each bond type"

        # printout("global max cutoff (angstrom)", max(rcutfac))
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
            ccs = wigner_coupling.get_coupling(
                self.__precomputed_wigner_3j,
                ldict,
                L_R=self.parameters.ace_L_R,
            )

        elif self.parameters.ace_coupling_type == "cg":
            ccs = cg_coupling.get_coupling(
                self.__precomputed_clebsch_gordan,
                ldict,
                L_R=self.parameters.ace_L_R,
            )

        else:
            sys.exit(
                "Coupling type "
                + str(self.parameters.ace_coupling_type)
                + " not recongised"
            )

        nus, limit_nus = self.calc_limit_nus()

        if not self.parameters.ace_types_like_snap:
            # NOTE to use unique ACE types for gridpoints, we must subtract off
            #  dummy descriptor counts (for non-grid element types)
            self.fingerprint_length = (
                ncols0
                + len(limit_nus)
                - (len(self.parameters.ace_elements) - 1)
            )
            # permutation symmetry adapted ACE labels
            Apot = AcePot(
                self.parameters.ace_elements,
                self.parameters.ace_reference_ens,
                self.parameters.ace_ranks,
                self.parameters.ace_nmax,
                self.parameters.ace_lmax,
                self.parameters.ace_nradbase,
                rcutfac,
                lmbda,
                rcinner,
                drcinner,
                lmin=self.parameters.ace_lmin,
                **{"input_nus": limit_nus, "ccs": ccs[self.parameters.ace_M_R]}
            )

            Apot.set_funcs(nulst=limit_nus, muflg=True, print_0s=True)
            return Apot.write_pot("coupling_coefficients")

        # add case for full basis separately
        # NOTE that this basis evaluates grid-atom and atom-atom descriptors that are not needed in the current implementation of MALA
        elif self.parameters.ace_types_like_snap:
            self.fingerprint_length = ncols0 + len(nus)
            # permutation symmetry adapted ACE labels
            Apot = AcePot(
                self.parameters.ace_elements,
                self.parameters.ace_reference_ens,
                self.parameters.ace_ranks,
                self.parameters.ace_nmax,
                self.parameters.ace_lmax,
                self.parameters.ace_nradbase,
                rcutfac,
                lmbda,
                rcinner,
                drcinner,
                lmin=self.parameters.ace_lmin,
                **{"input_nus": nus, "ccs": ccs[self.parameters.ace_M_R]}
            )
            return Apot.write_pot("coupling_coefficients")

    def calc_limit_nus(self):
        # TODO: Add documentation what this piece of code does.
        # Also, maybe make it private?
        ranked_chem_nus = []
        for ind, rank in enumerate(self.parameters.ace_ranks):
            rank = int(rank)
            PA_lammps, not_compat = acu.pa_labels_raw(
                rank,
                int(self.parameters.ace_nmax[ind]),
                int(self.parameters.ace_lmax[ind]),
                int(self.parameters.ace_mumax),
                lmin=int(self.parameters.ace_lmin[ind]),
            )
            ranked_chem_nus.append(PA_lammps)

        nus_unsort = [item for sublist in ranked_chem_nus for item in sublist]
        nus = nus_unsort.copy()
        mu0s = []
        mus = []
        ns = []
        ls = []
        for nu in nus_unsort:
            mu0ii, muii, nii, lii = acu.get_mu_n_l(nu)
            mu0s.append(mu0ii)
            mus.append(tuple(muii))
            ns.append(tuple(nii))
            ls.append(tuple(lii))
        nus.sort(key=lambda x: mus[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: ns[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: ls[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: mu0s[nus_unsort.index(x)], reverse=False)
        nus.sort(key=lambda x: len(x), reverse=False)
        nus.sort(key=lambda x: mu0s[nus_unsort.index(x)], reverse=False)
        musins = range(len(self.parameters.ace_elements) - 1)
        all_funcs = {}
        if self.parameters.ace_types_like_snap:
            byattyp, byattypfiltered = self.srt_by_attyp(nus, 1)
            if self.parameters.ace_grid_filter:
                assert (
                    self.parameters.ace_padfunc
                ), "must pad with at least 1 other basis function for other element types to work in LAMMPS - set padfunc=True"
                limit_nus = byattypfiltered["%d" % 0]
                if self.parameters.ace_padfunc:
                    for muii in musins:
                        limit_nus.append(byattypfiltered["%d" % muii][0])
            elif not self.parameters.ace_grid_filter:
                limit_nus = byattyp["%d" % 0]

        else:
            byattyp, byattypfiltered = self.srt_by_attyp(
                nus, len(self.parameters.ace_elements)
            )
            if self.parameters.ace_grid_filter:
                limit_nus = byattypfiltered[
                    "%d" % (len(self.parameters.ace_elements) - 1)
                ]
                assert (
                    self.parameters.ace_padfunc
                ), "must pad with at least 1 other basis function for other element types to work in LAMMPS - set padfunc=True"
                if self.parameters.ace_padfunc:
                    for muii in musins:
                        limit_nus.append(byattypfiltered["%d" % muii][0])
            elif not self.parameters.ace_grid_filter:
                limit_nus = byattyp[
                    "%d" % (len(self.parameters.ace_elements) - 1)
                ]
                if self.parameters.ace_padfunc:
                    for muii in musins:
                        limit_nus.append(byattyp["%d" % muii][0])
        # printout(
        #    "all basis functions", len(nus), "grid subset", len(limit_nus)
        # )

        return nus, limit_nus

    def get_default_settings(self):
        # TODO: Add documentation. Also, maybe make it private?
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
        # TODO: Add documentation. Also, maybe make it private?
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
            printout(
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
            printout(
                "NOTE: using dummy VDW radius of 2* covalent radius for %s"
                % elm2
            )
            vdwr2 = 2 * covr2
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

    def srt_by_attyp(self, nulst, remove_type=2):
        # TODO: Add documentation. Also, maybe make it private?
        mu0s = []
        for nu in nulst:
            mu0 = nu.split("_")[0]
            if mu0 not in mu0s:
                mu0s.append(mu0)
        mu0s = sorted(mu0s)
        byattyp = {mu0: [] for mu0 in mu0s}
        byattypfiltered = {mu0: [] for mu0 in mu0s}
        mumax = remove_type - 1
        for nu in nulst:
            mu0 = nu.split("_")[0]
            byattyp[mu0].append(nu)
            mu0ii, muii, nii, lii = acu.get_mu_n_l(nu)
            if mumax not in muii:
                byattypfiltered[mu0].append(nu)
        return byattyp, byattypfiltered

    def __init_wigner_3j(self, lmax):
        # TODO: Add documentation. Also, maybe make it private?
        # returns dictionary of all cg coefficients to be used at a given value of lmax
        try:
            with open(
                "%s/wig.pkl" % os.path.dirname(os.path.abspath(__file__)), "rb"
            ) as readinwig:
                cg = pickle.load(readinwig)
        except FileNotFoundError:
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
                                    cg[key] = self.__precomputed_wigner_3j(
                                        l1, m1, l2, m2, l3, m3
                                    )
            with open(
                "%s/wig.pkl" % os.path.dirname(os.path.abspath(__file__)), "wb"
            ) as writewig:
                pickle.dump(cg, writewig)
        return cg

    def __init_clebsch_gordan(self, lmax):
        # TODO: Add documentation. Also, maybe make it private?
        # returns dictionary of all cg coefficients to be used at a given value of lmax
        try:
            with open(
                "%s/cg.pkl" % os.path.dirname(os.path.abspath(__file__)), "rb"
            ) as readincg:
                cg = pickle.load(readincg)
        except FileNotFoundError:
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
            with open(
                "%s/cg.pkl" % os.path.dirname(os.path.abspath(__file__)), "wb"
            ) as writecg:
                pickle.dump(cg, writecg)
            # pickle.dump(cg,'cg.pkl')
        return cg

    @staticmethod
    def __init_element_lists():
        # Access periodic table data from mendeleev.
        element_info = fetch_table("elements")[
            ["symbol", "group_id", "atomic_radius"]
        ]
        element_info.set_index("symbol", inplace=True)

        # These ionic (actually, atomic radii) are taken from mendeleev.
        # In the original implementation of the ACE class, these were
        # hardcoded, now we get them automatically from mendeleev.
        ionic_radii = element_info.to_dict()["atomic_radius"]
        ionic_radii = {key: v / 100 for (key, v) in ionic_radii.items()}
        ionic_radii["G"] = 1.35

        # The metal list that was originally here only contains elements up to
        # the 13th group (I am not sure why, e.g., Al is excluded).
        # Additionally, only _some_ Lanthanoids and Actininoids were included,
        # but that may be due to the list having been outdated, since the old
        # name for Db (Ha) had been used. The code here now includes all
        # elements up to group 13 (minus H) and all Lanthanoids and
        # Actininoids.
        main_groups = element_info["group_id"] < 13.0
        actininoids_lanthanoids = element_info["group_id"].isna()
        main_groups = main_groups.to_dict()
        main_groups = [key for (key, v) in main_groups.items() if v]
        actininoids_lanthanoids = actininoids_lanthanoids.to_dict()
        actininoids_lanthanoids = [
            key for (key, v) in actininoids_lanthanoids.items() if v
        ]
        metal_list = list(set(main_groups + actininoids_lanthanoids))
        metal_list.remove("H")

        return ionic_radii, metal_list

    def clebsch_gordan(self, j1, m1, j2, m2, j3, m3):
        # TODO: Add documentation. Also, maybe make it private?
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

    def _read_feature_dimension_from_json(self, json_dict):
        nus, limit_nus = self.calc_limit_nus()

        if self.parameters.descriptors_contain_xyz:
            return len(limit_nus) - (len(self.parameters.ace_elements) - 1)
        else:
            return len(limit_nus) - (len(self.parameters.ace_elements) - 1) + 3
