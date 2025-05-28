"""ACE descriptor class."""

import glob
import os
import itertools
import sys
import pickle

import ase
import ase.io
from ase.data import atomic_numbers, atomic_masses
from importlib.util import find_spec
import numpy as np
from scipy import special
from mendeleev.fetch import fetch_table

from mala.common.parallelizer import printout
from mala.descriptors.lammps_utils import extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.descriptors.acelib.ace_potential import ACEPotential
import mala.descriptors.acelib.coupling_utils as ace_coupling_utils
import mala.descriptors.acelib.wigner_coupling as wigner_coupling
import mala.descriptors.acelib.clebsch_gordan_coupling as cg_coupling


class ACE(Descriptor):
    """Class for calculation and parsing of ACE descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.

    Attributes
    ----------
    couplings_yace_file : str
        File which holds the coupling coefficients. Can be provided by users,
        in which case consistency will be checked. If no file is detected, a
        new file is computed.
    """

    def __init__(self, parameters):
        super(ACE, self).__init__(parameters)

        # Initialize a dictionary with ionic radii (in Angstrom).
        # and a list containing all elements which are considered metals.
        self._atomic_radii, self._metal_list = self.__init_element_lists()

        # Initialize several arrays that are computed using
        # nested for-loops but can be reused.
        # Within these routines, these arrays are saved to pkl files
        # in the directory where ace.py is located.
        # If a pkl file is detected, the array is loaded from the pkl file,
        # otherwise it is computed and saved to a pkl file.
        # The arrays are used within the ACE descriptor calculation.
        # Which arrays are actually needed depends on which type of
        # coefficients are used for the ACE descriptor calculation.
        if self.parameters.ace_coupling_coefficients_type == "wigner3j":
            self.__precomputed_wigner_3j = self.__init_wigner_3j(
                self.parameters.ace_coupling_coefficients_maximum_l
            )
        if self.parameters.ace_coupling_coefficients_type == "clebsch_gordan":
            self.__precomputed_clebsch_gordan = self.__init_clebsch_gordan(
                self.parameters.ace_coupling_coefficients_maximum_l
            )

        # File which holds the coupling coefficients.
        # Can be provided by users, in which case consistency will be checked.
        # If no file is detected, a new file is computed.
        self.couplings_yace_file = None

        # Will get filled once the atoms object has been loaded.
        # mumax is the maximum chemical basis index (should be equal to the number of chemical types for now)
        self.__ace_mumax = None
        # constant energy to shift ace model by - should be 0/NONE in MALA
        self.__ace_reference_energy = None

        # Will get filled during calculation of coupling coefficients.
        # nus is the full list of descriptor labels while limit_nus is the list of grid-centered descriptors only (without grid-grid interactions)
        # bonds hold?
        self.__bonds = None
        self._maximum_cutoff_factor = None
        self._cutoff_factors = None
        self.__lambdas = None
        self.__nus = None
        self.__limit_nus = None

        # These used to live in the parameters subclass for the descriptors,
        # but I have moved them here, because I think they are defaults that
        # barely ever change. I will not be removing them outright, because
        # they may be needed for debugging.
        self.__ace_grid_filter = True
        self.__ace_padfunc = True

        # TODO: Move that into the ace.py class. Rename them, also they are
        # always 0 for now
        # Equivariance options for the ACE descriptors.
        # L_R: Transformation characteristics of the descriptor
        # (0: scalar, 1: vector, 2: tensor, etc.)
        # M_R: Selects which transformation within the subclass of
        # transformations selected by L_R.

        # Transformation characteristic and selector, also referred to as
        # L_R and M_R, respectively. The former governs which type of
        # transformation characteristic the descriptors follow (i.e., do they
        # transform as 0: scalar, 1: vector, 2: tensor, etc.). The latter
        # selects which transformation within the subclass of transformations
        # selected by L_R is used. E.g., if _transformation_characteristic=1,
        # then the descriptors may transform like a p_x, p_y, or p_z orbital,
        # depending on the value of _transformation_selector (-1, 0, 1).
        # For now, these will always be 0. I put them as class attributes
        # so that they may be changed for debugging purposes.
        self._transformation_characteristic = 0
        self._transformation_selector = 0

        # Consistency checks for the ACE settings.
        if (
            len(self.parameters.ace_included_expansion_ranks)
            != len(self.parameters.ace_maximum_l_per_rank)
            or len(self.parameters.ace_included_expansion_ranks)
            != len(self.parameters.ace_maximum_n_per_rank)
            or len(self.parameters.ace_included_expansion_ranks)
            != len(self.parameters.ace_minimum_l_per_rank)
        ):
            raise Exception(
                "ACE ranks, lmax, nmax, and lmin must have the same length"
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
        """
        Perform ACE descriptor calculation.

        This function is the main entry point for the calculation of the
        bispectrum descriptors. Currently, only LAMMPS is implemented.

        Parameters
        ----------
        outdir : string
            Path to the output directory.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ace_descriptors_np : numpy.ndarray
            The calculated bispectrum descriptors.
        """
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

        Parameters
        ----------
        outdir : string
            Path to the output directory.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ace_descriptors_np : numpy.ndarray
            The calculated bispectrum descriptors.
        """
        # For version compatibility; older lammps versions (the serial version
        # we still use on some machines) have these constants as part of the
        # general LAMMPS import.
        from lammps import constants as lammps_constants

        use_fp64 = kwargs.get("use_fp64", False)
        keep_logs = kwargs.get("keep_logs", False)

        lammps_format = "lammps-data"
        self.setup_lammps_tmp_files("acegrid", outdir)

        self.__init_element_arrays()

        ase.io.write(
            self._lammps_temporary_input, self._atoms, format=lammps_format
        )

        nx = self.grid_dimensions[0]
        ny = self.grid_dimensions[1]
        nz = self.grid_dimensions[2]

        # Calculate and store the coupling coefficients, if necessary.
        # If coupling coefficients are provided by the user, these are
        # first checked for consistency.
        # The bonds and cutoff have to always be computed, the coupling
        # coefficients only, if none were provided.
        self._calculate_bonds_and_cutoff()
        self.couplings_yace_file = self.check_coupling_coeffs(
            self.couplings_yace_file
        )
        if self.couplings_yace_file is None:
            self.couplings_yace_file = self._calculate_coupling_coeffs()

        # Create LAMMPS instance.
        lammps_dict = {
            "ace_coeff_file": self.couplings_yace_file,
            "rcutfac": self._maximum_cutoff_factor,
        }
        for idx, element in enumerate(sorted(list(set(self._atoms.numbers)))):
            lammps_dict["mass" + str(idx + 1)] = atomic_masses[element]

        lmp = self._setup_lammps(nx, ny, nz, lammps_dict)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        if self.parameters.custom_lammps_compute_file != "":
            lammps_compute_file = self.parameters.custom_lammps_compute_file
        else:
            filepath = os.path.join(os.path.dirname(__file__), "inputfiles")
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    lammps_compute_file = os.path.join(
                        filepath,
                        "in.acegridlocal_n{0}.python".format(
                            len(set(self._atoms.numbers))
                        ),
                    )
                else:
                    lammps_compute_file = os.path.join(
                        filepath,
                        "in.acegridlocal_defaultproc_n{0}.python".format(
                            len(set(self._atoms.numbers))
                        ),
                    )
            else:
                lammps_compute_file = os.path.join(
                    filepath,
                    "in.acegrid_n{0}.python".format(
                        len(set(self._atoms.numbers))
                    ),
                )

        # Do the LAMMPS calculation.
        lmp.file(lammps_compute_file)

        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full array at
        # the end of this function.
        # This is not necessarily true for the parallel case.
        if self.parameters._configuration["mpi"]:
            nrows_local = extract_compute_np(
                lmp,
                "agridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_ROWS,
            )
            ncols_local = extract_compute_np(
                lmp,
                "agridlocal",
                lammps_constants.LMP_STYLE_LOCAL,
                lammps_constants.LMP_SIZE_COLS,
            )
            if ncols_local != self.feature_size + 3:
                self.feature_size = ncols_local - 3
                # raise Exception("Inconsistent number of features.")

            ace_descriptors_np = extract_compute_np(
                lmp,
                "agridlocal",
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
                "agrid",
                0,
                2,
                (nz, ny, nx, self.feature_size),
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
        """
        Check the coupling coefficients for consistency.

        This reads the first line of the coupling coefficients file, which
        contains a list of the elements the coupling coefficients have been
        computed for, and checks whether this is consistent with the
        elements for this calculation.

        Parameters
        ----------
        coupling_file : str
            Path to the coupling coefficients file.

        Returns
        -------
        coupling_file : str
            Path to the coupling coefficients file. None if the file was
            found incompatible.
        """
        if coupling_file is not None and os.path.isfile(coupling_file):
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

                # We don't have to worry about order in this comparison,
                # because both lists are ordered, and the element lists
                # in the coupling coefficients are always saved in an
                # ordered fashion.
                compatible = True
                if element_list[-1] != "G":
                    compatible = False
                list_ase = sorted(list(set(self._atoms.symbols)))
                for element_idx in range(len(list_ase)):
                    if list_ase[element_idx] not in element_list:
                        compatible = False
                        break

                if compatible:
                    return coupling_file
                else:
                    return None
        else:
            return None

    def _calculate_bonds_and_cutoff(self):
        """
        Calculate bonds, cutoff radii, and other radial basis parameters.

        This is essentially part of the computation of the coupling
        coefficients, but a part that has to be done every time,
        since it initializes the cutoff radii.

        Radial basis functions in ACE descriptors are defined per
        bond type. "Bond type"s in ace descriptors are defined by all
        possible combinations of element types. These may be found with
        with itertools.product(ace_types,ace_types). For MALA, ace_types
        includes the gridpoints as a type, and typically descriptors are only
        used that form "bonds" with the gridpoints at the center.

        The cutoff radii determine how far from the gridpoint to search for
        atoms when calculating the grid-centered ACE descriptors. These are,
        by default, determined based on the atomic radii of the elements
        allowed in the system and must be defined per "bond type".

        The radial basis in the LAMMPS-based ACE descriptors use a scaled
        distance input rather than the radial distance between gridpoint and
        atom directly. This scaled distance, defined in Drautz 2019, allows
        one to make the ACE descriptors (exponentially) more sensitive to short
        distances from the grid center. The exponential scaling factor, lambda,
        must be defined per "bond type". A larger `lambda` provides more
        sensitivity for small distances compared to larger distances, and does
        so through an exponential function. As `lambda` approaches 0, the scaled
        distance approaches the true separation distance between the gridpoint
        and the atom `r`. The default values for `lambda` are chosen to be small
        fractions of the respective cutoff radii, so that small and large
        separations are given similar sensitivities. In practice, these are
        hyperparameters that should be optimized.

        """
        ncols0 = 3

        # "G" for grid has to be added to the element list.
        element_list = sorted(list(set(self._atoms.symbols))) + ["G"]

        # Bond types correspond to all bonds between elements.
        self.__bonds = [
            p for p in itertools.product(element_list, element_list)
        ]

        # Default settings for the cutoff radii and lambda values.
        (
            rc_range,
            rc_default,
            lmb_default,
        ) = self.__default_settings()

        self._cutoff_factors = [float(k) for k in rc_default.split()[2:]]
        self._maximum_cutoff_factor = np.max(self._cutoff_factors)
        self.__lambdas = [float(k) for k in lmb_default.split()[2:]]
        assert len(self.__bonds) == len(self._cutoff_factors) and len(
            self.__bonds
        ) == len(
            self.__lambdas
        ), """
            you must have rcutfac and lmbda defined for each bond type
            """
        self.__nus, self.__limit_nus = self.__calculate_limit_nus()

        # NOTE to use unique ACE types for gridpoints, we must subtract off
        #  dummy descriptor counts (for non-grid element types)
        self.feature_size = (
            ncols0 + len(self.__limit_nus) - (len(element_list) - 1)
        )

    def _calculate_coupling_coeffs(self):
        """
        Calculate generalized coupling coefficients and save them to file.

        By default, it will enumerate ACE descriptors according to the
        Permutation-adapted approach described in https://doi.org/10.1016/j.jcp.2024.113073.

        Returns
        -------
        coupling_file : str
            Path to the coupling coefficients file.
        """
        # "G" for grid has to be added to the element list.
        element_list = sorted(list(set(self._atoms.symbols))) + ["G"]

        # rcinner: Inner cutoff to turn on soft-core repulsions. This parameter should be 0.0 (OFF) for each bond type in MALA.
        # drcinner: Parameter for soft-core repulsions. This parameter should not matter if rcutinner is 0.0 (OFF) for each bond type in MALA.
        rcinner = [0.0] * len(self.__bonds)
        drcinner = [0.0] * len(self.__bonds)
        ldict = {
            ranki: li
            for ranki, li in zip(
                self.parameters.ace_included_expansion_ranks,
                self.parameters.ace_maximum_l_per_rank,
            )
        }

        # Coupling type: type of coupling coefficients used to reduce prodects of spherical harmonics (e.g., generalized wigner symbols or generalized clebsch-gordan coefficients)
        if self.parameters.ace_coupling_coefficients_type == "wigner3j":
            ccs = wigner_coupling.wigner_3j_coupling(
                self.__precomputed_wigner_3j,
                ldict,
                L_R=self._transformation_characteristic,
            )

        elif (
            self.parameters.ace_coupling_coefficients_type == "clebsch_gordan"
        ):
            ccs = cg_coupling.clebsch_gordan_coupling(
                self.__precomputed_clebsch_gordan,
                ldict,
                L_R=self._transformation_characteristic,
            )

        else:
            raise Exception(
                "Coupling type "
                + str(self.parameters.ace_coupling_coefficients_type)
                + " not recongised"
            )

        # Calculating permutation symmetry adapted ACE labels.
        # cutoff_factors : radial basis function cutoff per bond type. For example, if elements are ['Al','G'] then rcut must be supplied for each:(Al,Al)(Al,G)(G,Al)(G,G)
        Apot = ACEPotential(
            element_list,
            self.__ace_reference_energy,
            self.parameters.ace_included_expansion_ranks,
            self.parameters.ace_maximum_n_per_rank,
            self.parameters.ace_maximum_l_per_rank,
            max(self.parameters.ace_maximum_n_per_rank),
            self._cutoff_factors,
            self.__lambdas,
            ccs[self._transformation_selector],
            rcutinner=rcinner,
            drcutinner=drcinner,
            lmin=self.parameters.ace_minimum_l_per_rank,
        )

        # set the ace descriptor labels in the ACE "potential" class, to be read by LAMMPS as coupling coefficients
        # these are the descriptors calculated in lammps
        Apot.set_funcs(nulst=self.__limit_nus, print_0s=True)

        # Write the coupling coefficients to file.
        return Apot.write_pot(
            "coupling_coefficients_" + str.join("_", element_list[:-1])
        )

    def __calculate_limit_nus(self):
        """
        Enumerate ACE descriptors. By default, it will enumerate ACE descriptors according to the permutation-adapted approach described in https://doi.org/10.1016/j.jcp.2024.113073. After enumerating the complete set of atom-atom, atom-grid, grid-atom, and grid-grid descriptors, it will select the subset of grid-atom descriptors needed for MALA and will return those. It will include one dummy descriptor for each atom type as well (needed to make LAMMPS run).

        Returns
        -------
        nus : List
            Complete list of ACE descriptor labels in FitSNAP/LAMMPS format
            mu0_mu1,mu2...muN,n1,n2...nN,l1,l2...lN_L1-L2-...-L_{N-3}-L_{N-2}
            That include atom-atom, atom-grid, grid-atom, and grid-grid descriptors.

        limit_nus : List
            List of grid-atom descriptors (grid at center) that are used as input in MALA along with dummy atom-atom descriptors to make LAMMPS run.
        """
        # enumerate all descriptors, whether or not the chemical basis includes grid points (e.g., grid-grid interactions)
        ranked_chem_nus = []
        for ind, rank in enumerate(
            self.parameters.ace_included_expansion_ranks
        ):
            rank = int(rank)
            PA_lammps = ace_coupling_utils.compute_pa_labels_raw(
                rank,
                int(self.parameters.ace_maximum_n_per_rank[ind]),
                int(self.parameters.ace_maximum_l_per_rank[ind]),
                int(self.__ace_mumax),
                lmin=int(self.parameters.ace_minimum_l_per_rank[ind]),
            )
            ranked_chem_nus.append(PA_lammps)

        # sort the descriptors according to chemical function indices, radial function indices, and angular function indices
        nus_unsort = [item for sublist in ranked_chem_nus for item in sublist]
        nus = nus_unsort.copy()
        mu0s = []
        mus = []
        ns = []
        ls = []
        for nu in nus_unsort:
            mu0ii, muii, nii, lii = ace_coupling_utils.calculate_mu_n_l(nu)
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
        musins = range(len(list(set(self._atoms.symbols))))

        # collect the descriptors according to central atom type (one entry here will correspond to grid-centered descriptors used in MALA)
        # +1 here because this should include the G for grid.
        byattyp, byattypfiltered = self.__sort_by_atom_type(
            nus, len(list(set(self._atoms.symbols))) + 1
        )
        if self.__ace_grid_filter:
            limit_nus = byattypfiltered[
                "%d" % (len(list(set(self._atoms.symbols))))
            ]
            # assert that dummy descriptor must be added per atom type to make LAMMPS run
            assert self.__ace_padfunc, (
                "Must pad with at least 1 other basis function for other "
                "element types to work in LAMMPS - set padfunc=True"
            )
            if self.__ace_padfunc:
                for muii in musins:
                    limit_nus.append(byattypfiltered["%d" % muii][0])
        # if not using grid-grid and atom-atom descriptors, the dummy function is not strictly required
        elif not self.__ace_grid_filter:
            limit_nus = byattyp["%d" % (len(list(set(self._atoms.symbols))))]
            if self.__ace_padfunc:
                for muii in musins:
                    limit_nus.append(byattyp["%d" % muii][0])

        return nus, limit_nus

    def __default_settings(self):
        """
        Based on atomic radii, automatically generate the radial cutoffs and other ACE hyperparamters. These may need to be adjusted in practice for an optimal model, but these default values are often reasonable. These settings are generated for a provided list of ace_elements. These element symbols must include 'G' for the grid points. In ACE, all possible combinations of these elements determine the "bond types" spanned by the ACE chemical basis (the chemical basis is the delta function basis used in Drautz 2019). For example, the "bond types" resulting from all possible combinations of ['Al','G'] are determined with :code:`itertools.product()` in python. They are (Al,Al)(Al,G)(G,Al)(G,G). Default hyperparameters are defined in this function for all of the "bond types", even though or mala, only those of type (G,X) are kept.

        Returns
        -------
        rc_range : dict
            Dictionary with bond types as keys and cutoff radii ranges as
            values.

        rc_def_str : str
            String with default cutoff radii values.

        lmb_def_str : str
            String with default lambda values.
        """
        # for each bond type, determine what cutoff radii should be allowed
        # these values (and ranges of values) are determined based on the element type with some defaults provided below
        rc_range = {bp: None for bp in self.__bonds}
        rin_def = {bp: None for bp in self.__bonds}
        rc_def = {bp: None for bp in self.__bonds}
        fac_per_elem = {key: 0.5 for key in list(self._atomic_radii.keys())}
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

        for bond in self.__bonds:
            rcmini, rcmaxi = self.__default_cutoff_factors(bond)
            defaultri = (rcmaxi + rcmini) / np.sum(
                [fac_per_elem[elem] * 2 for elem in bond]
            )
            defaultrcinner = (
                rcmini
            ) * 0.25  # 1/4 of shortest ionic bond length
            rc_range[bond] = [rcmini, rcmaxi]
            rc_def[bond] = defaultri * self.parameters.ace_cutoff_factor
            rin_def[bond] = defaultrcinner
        # shift = ( max(rc_def.values()) - min(rc_def.values())  )/2
        shift = np.std(list(rc_def.values()))
        # if apply_shift:
        for bond in self.__bonds:
            if rc_def[bond] != max(rc_def.values()):
                if self.parameters.ace_balance_cutoff_radii_for_elements:
                    rc_def[bond] = rc_def[bond] + shift  # *nshell
                else:
                    rc_def[bond] = rc_def[bond]  # *nshell
        default_lmbs = [i * 0.05 for i in list(rc_def.values())]
        rc_def_lst = ["%1.3f"] * len(self.__bonds)
        rc_def_str = "rcutfac = " + "  ".join(b for b in rc_def_lst) % tuple(
            list(rc_def.values())
        )
        lmb_def_lst = ["%1.3f"] * len(self.__bonds)
        lmb_def_str = "lambda = " + "  ".join(b for b in lmb_def_lst) % tuple(
            default_lmbs
        )
        return rc_range, rc_def_str, lmb_def_str

    def __default_cutoff_factors(self, elms):
        """
        Compute default cutoff factors.

        It is assumed that most important interactions will occur within ~2r_atom (about two neighbor shells) for a given atom type. This is used to specify the bounds for the default cutoffs.

        Parameters
        ----------
        elms

        Returns
        -------
        rcmini : float
            Minimum cutoff factor.

        rcmaxi : float
            Maximum cutoff factor.
        """
        # set default cutoffs given an atom type with the other (defaults to use Gold as the atom type for grid points which offers reasonably large cutoff)
        elms = sorted(elms)
        elm1, elm2 = tuple(elms)
        try:
            atnum1 = ase.data.atomic_numbers[elm1]
        except KeyError:
            atnum1 = ase.data.atomic_numbers["Au"]
        covr1 = ase.data.covalent_radii[atnum1]
        vdwr1 = ase.data.vdw_radii[atnum1]
        ionr1 = self._atomic_radii[elm1]
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
        ionr2 = self._atomic_radii[elm2]
        if np.isnan(vdwr2):
            printout(
                "NOTE: using dummy VDW radius of 2* covalent radius for %s"
                % elm2
            )
            vdwr2 = 2 * covr2
        minbond = ionr1 + ionr2
        if self.parameters.ace_larger_cutoff_for_metals:
            if elm1 not in self._metal_list and elm2 not in self._metal_list:
                maxbond = vdwr1 + vdwr2
            elif elm1 in self._metal_list and elm2 not in self._metal_list:
                maxbond = ionr1 + vdwr2
                minbond = (ionr1 + ionr2) * 0.8
            elif elm1 in self._metal_list and elm2 in self._metal_list:
                maxbond = ionr1 + ionr2
                minbond = (ionr1 + ionr2) * 0.8
            else:
                maxbond = ionr1 + ionr2
        else:
            maxbond = vdwr1 + vdwr2
        midbond = (maxbond + minbond) / 2
        # by default, return the ionic bond length * number of bonds for
        # minimum
        returnmin = minbond
        if self.parameters.ace_use_maximum_cutoff_per_element:
            # return vdw bond length if requested
            returnmax = maxbond
        else:
            # return hybrid vdw/ionic bonds by default
            returnmax = midbond
        return round(returnmin, 3), round(returnmax, 3)

    @staticmethod
    def __sort_by_atom_type(nulst, remove_type=2):
        """
        Sorts collections of ace descriptors by the element type (which is encoded by the mu0 chemical funciton index).

        This is useful for organizing the descriptor matrix returned from LAMMPs. All descriptors used as input for MALA are given first.

        Parameters
        ----------
        nulst : List
            List of ace descripor labels in LAMMPS/FitSNAP format.

        remove_type : int
            Grid index (to remove from chemical basis) if it were left in, descriptors would contain mixed grid-grid and grid-atom interactions. It defaults to 2 to remove the grid type for a single element system. In general this value should be equal to the number of elements + the grid type. For example ['Ga',N','G'], remove_type should be 3.

        Returns
        -------
        byattyp : dict
            Complete set of atom-atom, atom-grid, grid-atom, and grid-grid descriptors

        byattypfiltered : dict
            Descriptors with grid-grid interactions filtered out.
        """
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
            mu0ii, muii, nii, lii = ace_coupling_utils.calculate_mu_n_l(nu)
            if mumax not in muii:
                byattypfiltered[mu0].append(nu)
        return byattyp, byattypfiltered

    def __init_wigner_3j(self, lmax):
        """
        Compute Wigner 3j coefficients.

        Returns dictionary of all cg coefficients to be used at a given value
        of lmax. Reads these coefficients from a pickle file, if one is found,
        and computes and stores them otherwise. Since these coefficients
        are the same across different calculations, they are stored in the
        MALA directory upon first execution of the code and can be reused
        essentially always.

        Parameters
        ----------
        lmax : int
            Maximum l value for precomputation.

        Returns
        -------
        wigner : dict
            Dictionary of all Wigner 3j coefficients to be used at a given
            value of lmax.
        """
        # We only recompute the coefficients if there is no file we can use.
        # A file we can use can either mean one for the exact same lmax or
        # one for a higher lmax.
        raw_list = glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "wig_*.pkl",
            )
        )
        lmaxes = [
            int(os.path.basename(x).split("_")[1].split(".")[0])
            for x in raw_list
        ]
        if len(lmaxes) > 0:
            loaded_lmax = max(lmax, min(lmaxes))
        else:
            loaded_lmax = lmax

        # We try to load the file and recompute, if necessary.
        file_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "wig_" + str(loaded_lmax) + ".pkl",
        )

        try:
            with open(file_name, "rb") as readinwig:
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
                                    cg[key] = self._wigner_3j(
                                        l1, m1, l2, m2, l3, m3
                                    )
            with open(file_name, "wb") as writewig:
                pickle.dump(cg, writewig)
        return cg

    def __init_clebsch_gordan(self, lmax):
        """
        Compute Clebsch-Gordan coefficients.

        Returns dictionary of all cg coefficients to be used at a given value
        of lmax. Reads these coefficients from a pickle file, if one is found,
        and computes and stores them otherwise. Since these coefficients
        are the same across different calculations, they are stored in the
        MALA directory upon first execution of the code and can be reused
        essentially always.

        Parameters
        ----------
        lmax : int
            Maximum l value for precomputation.

        Returns
        -------
        cg : dict
            Dictionary of all Clebsch-Gordan coefficients to be used at a given
            value of lmax.
        """
        # We only recompute the coefficients if there is no file we can use.
        # A file we can use can either mean one for the exact same lmax or
        # one for a higher lmax.
        raw_list = glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "cg_*.pkl",
            )
        )
        lmaxes = [
            int(os.path.basename(x).split("_")[1].split(".")[0])
            for x in raw_list
        ]
        if len(lmaxes) > 0:
            loaded_lmax = max(lmax, min(lmaxes))
        else:
            loaded_lmax = lmax

        # We try to load the file and recompute, if necessary.
        file_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cg_" + str(loaded_lmax) + ".pkl",
        )
        try:
            with open(file_name, "rb") as readincg:
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
                                    cg[key] = self._clebsch_gordan(
                                        l1, m1, l2, m2, l3, m3
                                    )
            with open(file_name, "wb") as writecg:
                pickle.dump(cg, writecg)
            # pickle.dump(cg,'cg.pkl')
        return cg

    def __init_element_arrays(self):
        """
        Initialize reference energy per atom type and max chemical basis index, mumax, for the system. These depend on the number of elements.

        Typically, one will set the ace reference energy to 0.0 eV for each element type. It is possible to use this to apply a constant shift to the energy if using a linear model.

        For standard usage, the maximum chemical basis index, mumax, should be 1 larger than the number of chemical species in the system. The + 1 is to account for the grid point as a unique "species" in the ACE descriptors.

        Parameters
        ----------
        None: None

        Returns
        -------
        ace_mumax : int
            Maximum chemical basis function index (should be equal to the number of species + 1 where the  +1 is for the grid point)

        ace_reference_energy : List
            The 0th order expansion term in a linear ACE model. Should be 0 for all element types in MALA.
        """
        self.__ace_mumax = len(list(set(self._atoms.symbols))) + 1
        self.__ace_reference_energy = [0.0] * self.__ace_mumax

    @staticmethod
    def __init_element_lists():
        # initially, the hard-coded values included vdW radii for elements where available (thus the weird definition of metal here). Where they werent available, other radii were used. We may use ionic radii instead, but we may want to change the factor by which we multiply the radii by to get the default cutoffs. Having them be ~2x the vdW radii is usually a good starting point.
        """
        Initialize a dictionary with ionic/vdW radii and list of metals.

        This function initializes a dictionary with ionic radii (in
        Angstrom) and a list containing all elements which are considered
        metals. Both are done via mendeleev. The definition of metal here is
        "everything in groups 12 and below, including Lanthanoids and
        Actininoids, excluding hydrogen". For the smaller elements, different
        default radii are beneficial.
        Initially, the hard-coded values included vdW radii for elements where
        available (thus the weird definition of metal here). Where they werent
        available, other radii were used. We may use ionic radii instead, but
        we may want to change the factor by which we multiply the radii by to
        get the default cutoffs. Having them be ~2x the vdW radii is usually a
        good starting point.


        Returns
        -------
        atomic_radii : dict
            Dictionary with ionic radii.

        metal_list : List
            List of metals.
        """
        # Access periodic table data from mendeleev.
        element_info = fetch_table("elements")[
            ["symbol", "group_id", "atomic_radius"]
        ]
        element_info.set_index("symbol", inplace=True)

        # These ionic (actually, atomic radii) are taken from mendeleev.
        # In the original implementation of the ACE class, these were
        # hardcoded, now we get them automatically from mendeleev.
        atomic_radii = element_info.to_dict()["atomic_radius"]
        atomic_radii = {key: v / 100 for (key, v) in atomic_radii.items()}
        atomic_radii["G"] = 1.35

        # The metal list that was originally here only contains elements up to
        # the 13th group (I am not sure why, e.g., Al is excluded).
        # Additionally, only _some_ Lanthanoids and Actinoids were included,
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

        return atomic_radii, metal_list

    @staticmethod
    def _clebsch_gordan(j1, m1, j2, m2, j3, m3):
        """
        Compute a traditional Clebsch-Gordan coefficient, used to reduce products of spherical harmonics occuring in ACE descriptors.

        Calculation is based on Eqs. 4-5 of: https://hal.inria.fr/hal-01851097/document .

        Parameters
        ----------
        j1 : int
            angluar momentum quantum number for spherical harmonic 1

        m1 : int
            projection quantum number for spherical harmonic 1

        j2: int
            angluar momentum quantum number for spherical harmonic 2

        m2: int
            projection quantum number for spherical harmonic 2

        j3: int
            angluar momentum quantum number for spherical harmonic 3

        m3: int
            projection quantum number for spherical harmonic 1

        Returns
        -------
        cg_coefficient : float
            Clebsch-Gordan coefficient for combination of j1,m1,j2,m2,j3,m3.
        """
        # CODE IS VERIFIED: test non-zero indices in Wolfram using format
        # ClebschGordan[{j1,m1},{j2,m2},{j3,m3}]
        # Rules:
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

            # k conditions (see eq.5 of
            # https://hal.inria.fr/hal-01851097/document)
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

    @staticmethod
    def _wigner_3j(j1, m1, j2, m2, j3, m3):
        """
        Compute a traditional Wigner 3j symbol, used to reduce products of spherical harmonics occuring in ACE descriptors.

        Uses relation between Clebsch-Gordann coefficients and Wigner 3j symbols to evaluate Wigner 3j symbols.

        Parameters
        ----------
        j1 : int
            angluar momentum quantum number for spherical harmonic 1

        m1 : int
            projection quantum number for spherical harmonic 1

        j2: int
            angluar momentum quantum number for spherical harmonic 2

        m2: int
            projection quantum number for spherical harmonic 2

        j3: int
            angluar momentum quantum number for spherical harmonic 3

        m3: int
            projection quantum number for spherical harmonic 1

        Returns
        -------
        wigner_3j_coefficient : float
            Wigner 3j coefficient for combination of j1,m1,j2,m2,j3,m3.
        """
        # CODE IS VERIFIED by wolframalpha.com
        cg = ACE._clebsch_gordan(j1, m1, j2, m2, j3, -m3)

        num = np.longdouble((-1) ** (j1 - j2 - m3))
        denom = np.longdouble(((2 * j3) + 1) ** (1 / 2))

        return cg * num / denom  # float(num/denom)

    def _read_feature_dimension_from_json(self, json_dict):
        """
        Read the feature dimension from a saved JSON file.

        For ACE descriptors, the feature dimension does directly depend on
        the number of atoms.

        Parameters
        ----------
        json_dict : dict
            Dictionary containing info loaded from the JSON file.
        """
        nus, limit_nus = self.__calculate_limit_nus()

        if self.parameters.descriptors_contain_xyz:
            return len(limit_nus) - (len(list(set(self._atoms.symbols))))
        else:
            return len(limit_nus) - (len(list(set(self._atoms.symbols)))) + 3
