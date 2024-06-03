"""ACE descriptor class."""

import os
import itertools

import ase
import ase.io

from importlib.util import find_spec
import numpy as np
from scipy.spatial import distance

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





