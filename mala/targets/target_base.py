"""Base class for all target calculators."""
from ase.units import Rydberg, Bohr, kB
import ase.io
import numpy as np
from mala.common.parameters import Parameters, ParametersTargets
from .calculation_helpers import fermi_function


class TargetBase:
    """
    Base class for all target quantity parser.

    Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file
    format and performs postprocessing calculations on the quantity.

    Parameters
    ----------
    params : mala.common.parameters.Parameters or
    mala.common.parameters.ParametersTargets
        Parameters used to create this TargetBase object.
    """

    def __init__(self, params):
        if isinstance(params, Parameters):
            self.parameters = params.targets
        elif isinstance(params, ParametersTargets):
            self.parameters = params
        else:
            raise Exception("Wrong type of parameters for Targets class.")
        self.target_length = 0
        self.fermi_energy_eV = None
        self.temperature_K = None
        self.grid_spacing_Bohr = None
        self.number_of_electrons = None
        self.band_energy_dft_calculation = None
        self.total_energy_dft_calculation = None
        self.grid_dimensions = [0, 0, 0]
        self.atoms = None
        self.qe_input_data = {
                "occupations": 'smearing',
                "calculation": 'scf',
                "restart_mode": 'from_scratch',
                "prefix": 'MALA',
                "pseudo_dir": self.parameters.pseudopotential_path,
                "outdir": './',
                "ibrav": None,
                "smearing": 'fermi-dirac',
                "degauss": None,
                "ecutrho": None,
                "ecutwfc": None,
                "nosym": True,
                "noinv": True,
        }

        # It has been shown that the number of k-points
        # does not really affect the QE post-processing calculation.
        # This is because we only evaulate density-dependent contributions
        # with QE. However, there were some (very small) inaccuracies when
        # operating only at the gamma point. Consequently, MALA defaults
        # to a small k-grid to ensure best accuracy and performance.
        # UPDATE 23.04.2021: As per discussion bewteen Normand Modine and Lenz
        # Fiedler, the number of k-points is moved to 1.
        # The small inaccuracies are neglected for now.
        self.kpoints = None  # (2, 2, 2)
        self.qe_pseudopotentials = {}

    @property
    def qe_input_data(self):
        """Input data for QE TEM calls."""

        # Update the pseudopotential path from Parameters.
        self._qe_input_data["pseudo_dir"] = \
            self.parameters.pseudopotential_path
        return self._qe_input_data

    @qe_input_data.setter
    def qe_input_data(self, value):
        self._qe_input_data = value

    def read_from_cube(self):
        """Read the quantity from a .cube file."""
        raise Exception("No function defined to read this quantity "
                        "from a .cube file.")

    def read_from_qe_dos_txt(self):
        """Read the quantity from a Quantum Espresso .dos.txt file."""
        raise Exception("No function defined to read this quantity "
                        "from a qe.dos.txt file")

    def get_density(self):
        """Get the electronic density."""
        raise Exception("No function to calculate or provide the "
                        "density has been implemented for this target type.")

    def get_density_of_states(self):
        """Get the density of states."""
        raise Exception("No function to calculate or provide the"
                        "density of states (DOS) has been implemented "
                        "for this target type.")

    def get_band_energy(self):
        """Get the band energy."""
        raise Exception("No function to calculate or provide the"
                        "band energy has been implemented for this target "
                        "type.")

    def get_number_of_electrons(self):
        """Get the number of electrons."""
        raise Exception("No function to calculate or provide the number of"
                        " electrons has been implemented for this target "
                        "type.")

    def get_total_energy(self):
        """Get the total energy."""
        raise Exception("No function to calculate or provide the number "
                        "of electons has been implemented for this target "
                        "type.")

    def read_additional_calculation_data(self, data_type, path_to_file=""):
        """
        Read in additional input about a calculation.

        This is e.g. necessary when we operate with preprocessed
        data for the training itself but want to take into account other
        physical quantities (such as the fermi energy or the electronic
        temperature) for post processing.

        Parameters
        ----------
        data_type : string
            Type of data or file that is used. Currently only supports
            qe.out for Quantum Espresso outfiles.

        path_to_file : string
            Path to the file that is used.
        """
        if data_type == "qe.out":
            # Reset everything.
            self.fermi_energy_eV = None
            self.temperature_K = None
            self.grid_spacing_Bohr = None
            self.number_of_electrons = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms = None

            # Read the file.
            self.atoms = ase.io.read(path_to_file, format="espresso-out")
            vol = self.atoms.get_volume()
            self.fermi_energy_eV = self.atoms.get_calculator().\
                get_fermi_level()

            # Parse the file for energy values.
            total_energy = None
            past_calculation_part = False
            bands_included = True
            with open(path_to_file) as out:
                pseudolinefound = False
                lastpseudo = None
                for line in out:
                    if "End of self-consistent calculation" in line:
                        past_calculation_part = True
                    if "number of electrons       =" in line:
                        self.number_of_electrons = np.float64(line.split('=')
                                                              [1])
                    if "Fermi-Dirac smearing, width (Ry)=" in line:
                        self.temperature_K = np.float64(line.split('=')[2]) * \
                                             Rydberg / kB
                    if "xc contribution" in line:
                        xc_contribution = float((line.split('=')[1]).
                                                split('Ry')[0])
                        break
                    if "one-electron contribution" in line:
                        one_electron_contribution = float((line.split('=')[1]).
                                                          split('Ry')[0])
                    if "hartree contribution" in line:
                        hartree_contribution = float((line.split('=')[1]).
                                                     split('Ry')[0])
                    if "FFT dimensions" in line:
                        dims = line.split("(")[1]
                        self.grid_dimensions[0] = int(dims.split(",")[0])
                        self.grid_dimensions[1] = int(dims.split(",")[1])
                        self.grid_dimensions[2] = int((dims.split(",")[2]).
                                                      split(")")[0])
                    if "bravais-lattice index" in line:
                        self.qe_input_data["ibrav"] = int(line.split("=")[1])
                    if "kinetic-energy cutoff" in line:
                        self.qe_input_data["ecutwfc"] \
                            = float((line.split("=")[1]).split("Ry")[0])
                    if "charge density cutoff" in line:
                        self.qe_input_data["ecutrho"] \
                            = float((line.split("=")[1]).split("Ry")[0])
                    if "smearing, width" in line:
                        self.qe_input_data["degauss"] \
                            = float(line.split("=")[-1])
                    if pseudolinefound:
                        self.qe_pseudopotentials[lastpseudo.strip()] \
                            = line.split("/")[-1].strip()
                        pseudolinefound = False
                        lastpseudo = None
                    if "PseudoPot." in line:
                        pseudolinefound = True
                        lastpseudo = (line.split("for")[1]).split("read")[0]
                    if "total energy" in line and past_calculation_part:
                        if total_energy is None:
                            total_energy \
                                = float((line.split('=')[1]).split('Ry')[0])
                    if "set verbosity='high' to print them." in line:
                        bands_included = False

            # Post process the text values.
            cell_volume = vol / (self.grid_dimensions[0] *
                                 self.grid_dimensions[1] *
                                 self.grid_dimensions[2] * Bohr ** 3)
            self.grid_spacing_Bohr = cell_volume ** (1 / 3)

            # Unit conversion
            self.total_energy_dft_calculation = total_energy*Rydberg

            # Calculate band energy, if the necessary data is included in
            # the output file.
            if bands_included:
                eigs = np.transpose(
                    self.atoms.get_calculator().band_structure().
                    energies[0, :, :])
                kweights = self.atoms.get_calculator().get_k_point_weights()
                eband_per_band = eigs * fermi_function(eigs,
                                                       self.fermi_energy_eV,
                                                       self.temperature_K)
                eband_per_band = kweights[np.newaxis, :] * eband_per_band
                self.band_energy_dft_calculation = np.sum(eband_per_band)
        else:
            raise Exception("Unsupported auxiliary file type.")

    def set_pseudopotential_path(self, newpath):
        """
        Set a path where your pseudopotentials are stored.

        This is needed for doing QE calculations.
        """
        self.qe_input_data["pseudo_dir"] = newpath

    def get_energy_grid(self):
        """Get energy grid."""
        raise Exception("No method implement to calculate an energy grid.")

    @staticmethod
    def convert_units(array, in_units="eV"):
        """
        Convert the units of an array into the MALA units.

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
        raise Exception("No unit conversion method implemented for"
                        " this target type.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

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
        raise Exception("No unit back conversion method implemented "
                        "for this target type.")

    def restrict_data(self, array):
        """
        Restrict target data to only contain physically meaningful values.

        For the LDOS this e.g. implies non-negative values. The type
        of data restriction is specified by the parameters.

        Parameters
        ----------
        array : numpy.array
            Numpy array, for which the restrictions are to be applied.

        Returns
        -------
        array : numpy.array
            The same array, with restrictions enforced.
        """
        if self.parameters.restrict_targets == "zero_out_negative":
            array[array < 0] = 0
            return array
        elif self.parameters.restrict_targets == "absolute_values":
            array[array < 0] *= -1
            return array
        elif self.parameters.restrict_targets is None:
            return array
        else:
            raise Exception("Wrong data restriction.")


