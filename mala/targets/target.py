"""Base class for all target calculators."""
from abc import ABC, abstractmethod

from ase.units import Rydberg, Bohr, kB
import ase.io
import numpy as np

from mala.common.parameters import Parameters, ParametersTargets
from mala.common.parallelizer import printout
from mala.targets.calculation_helpers import fermi_function


class Target(ABC):
    """
    Base class for all target quantity parser.

    Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file
    format and performs postprocessing calculations on the quantity.

    Parameters
    ----------
    params : mala.common.parameters.Parameters or
    mala.common.parameters.ParametersTargets
        Parameters used to create this Target object.
    """

    def __new__(cls, params: Parameters):
        """
        Create a Target instance.

        The correct type of target calculator will automatically be
        instantiated by this class if possible. You can also instantiate
        the desired target directly by calling upon the subclass.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this target calculator.
        """
        target = None

        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == Target:
            if isinstance(params, Parameters):
                targettype = params.targets.target_type
            elif isinstance(params, ParametersTargets):
                targettype = params.target_type
            else:
                raise Exception("Wrong type of parameters for Targets class.")

            if targettype == 'LDOS':
                from mala.targets.ldos import LDOS
                target = super(Target, LDOS).__new__(LDOS)
            if targettype == 'DOS':
                from mala.targets.dos import DOS
                target = super(Target, DOS).__new__(DOS)
            if targettype == 'Density':
                from mala.targets.density import Density
                target = super(Target, Density).__new__(Density)

            if target is None:
                raise Exception("Unsupported target parser/calculator.")
        else:
            target = super(Target, cls).__new__(cls)

        return target

    def __init__(self, params):
        if isinstance(params, Parameters):
            self.parameters = params.targets
        elif isinstance(params, ParametersTargets):
            self.parameters = params
        else:
            raise Exception("Wrong type of parameters for Targets class.")
        self.fermi_energy_eV = None
        self.temperature_K = None
        self.voxel_Bohr = None
        self.number_of_electrons = None
        self.number_of_electrons_from_eigenvals = None
        self.band_energy_dft_calculation = None
        self.total_energy_dft_calculation = None
        self.grid_dimensions = [0, 0, 0]
        self.atoms = None
        self.electrons_per_atom = None
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

    @abstractmethod
    def get_feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        pass

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

    def read_additional_calculation_data(self, data_type, data=""):
        """
        Read in additional input about a calculation.

        This is e.g. necessary when we operate with preprocessed
        data for the training itself but want to take into account other
        physical quantities (such as the fermi energy or the electronic
        temperature) for post processing.

        Parameters
        ----------
        data_type : string
            Type of data or file that is used. Currently supporte are:

            - "qe.out" : Read the additional information from a QuantumESPRESSO
              output file.

            - "atoms+grid" : Provide a grid and an atoms object from which to
              predict. Except for the number of electrons,
              this mode will not change any member variables;
              values have to be adjusted BEFORE.

        data : string or list
            Data from which additional calculation data is inputted.
        """
        if data_type == "qe.out":
            # Reset everything.
            self.fermi_energy_eV = None
            self.temperature_K = None
            self.voxel_Bohr = None
            self.number_of_electrons = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms = None

            # Read the file.
            self.atoms = ase.io.read(data, format="espresso-out")
            vol = self.atoms.get_volume()
            self.fermi_energy_eV = self.atoms.get_calculator().\
                get_fermi_level()

            # Parse the file for energy values.
            total_energy = None
            past_calculation_part = False
            bands_included = True
            with open(data) as out:
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

            # The voxel is needed for e.g. LDOS integration.
            self.voxel_Bohr = self.atoms.cell.copy()
            self.voxel_Bohr[0] = self.voxel_Bohr[0] / (
                        self.grid_dimensions[0] * Bohr)
            self.voxel_Bohr[1] = self.voxel_Bohr[1] / (
                        self.grid_dimensions[1] * Bohr)
            self.voxel_Bohr[2] = self.voxel_Bohr[2] / (
                        self.grid_dimensions[2] * Bohr)

            # This is especially important for size extrapolation.
            self.electrons_per_atom = self.number_of_electrons/len(self.atoms)

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
                enum_per_band = fermi_function(eigs,
                                               self.fermi_energy_eV,
                                               self.temperature_K)
                enum_per_band = kweights[np.newaxis, :] * enum_per_band
                self.number_of_electrons_from_eigenvals = np.sum(enum_per_band)

        elif data_type == "atoms+grid":
            # Reset everything that we can get this way.
            self.voxel_Bohr = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms: ase.Atoms = data[0]

            # Read the file.
            vol = self.atoms.get_volume()

            # Parse the file for energy values.
            self.grid_dimensions[0] = data[1][0]
            self.grid_dimensions[1] = data[1][1]
            self.grid_dimensions[2] = data[1][2]

            # The voxel is needed for e.g. LDOS integration.
            self.voxel_Bohr = self.atoms.cell.copy()
            self.voxel_Bohr[0] = self.voxel_Bohr[0] / (
                        self.grid_dimensions[0] * Bohr)
            self.voxel_Bohr[1] = self.voxel_Bohr[1] / (
                        self.grid_dimensions[1] * Bohr)
            self.voxel_Bohr[2] = self.voxel_Bohr[2] / (
                        self.grid_dimensions[2] * Bohr)

            if self.electrons_per_atom is None:
                printout("No number of electrons per atom provided, "
                         "MALA cannot guess the number of electrons "
                         "in the cell with this. Energy calculations may be"
                         "wrong.")

            else:
                self.number_of_electrons = self.electrons_per_atom *\
                                           len(self.atoms)

        else:
            raise Exception("Unsupported auxiliary file type.")

    def get_energy_grid(self):
        """Get energy grid."""
        raise Exception("No method implement to calculate an energy grid.")

    def get_real_space_grid(self):
        """Get the real space grid."""
        grid3D = np.zeros((self.grid_dimensions[0], self.grid_dimensions[1],
                           self.grid_dimensions[2], 3), dtype=np.float64)
        for i in range(0, self.grid_dimensions[0]):
            for j in range(0, self.grid_dimensions[1]):
                for k in range(0, self.grid_dimensions[2]):
                    grid3D[i, j, k, :] = np.matmul(self.voxel_Bohr, [i, j, k])
        return grid3D

    @staticmethod
    def write_tem_input_file(atoms_Angstrom, qe_input_data,
                             qe_pseudopotentials,
                             grid_dimensions, kpoints):
        """
        Write a QE-style input file for the total energy module.

        Usually, the used parameters should correspond to the properties of
        the object calling this function, but they don't necessarily have
        to.

        Parameters
        ----------
        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        grid_dimensions : list
            A list containing the x,y,z dimensions of the real space grid.

        kpoints : dict
            k-grid used, usually None or (1,1,1) for TEM calculations.
        """
        # Specify grid dimensions, if any are given.
        if grid_dimensions[0] != 0 and \
                grid_dimensions[1] != 0 and \
                grid_dimensions[2] != 0:
            qe_input_data["nr1"] = grid_dimensions[0]
            qe_input_data["nr2"] = grid_dimensions[1]
            qe_input_data["nr3"] = grid_dimensions[2]
            qe_input_data["nr1s"] = grid_dimensions[0]
            qe_input_data["nr2s"] = grid_dimensions[1]
            qe_input_data["nr3s"] = grid_dimensions[2]

        # Might be needed for test purposes, the Be2 test data
        # for example has symmetry, even though it was deactivated for
        # the DFT calculation. If symmetry is then on in here, that
        # leads to errors.
        # qe_input_data["nosym"] = False
        ase.io.write("mala.pw.scf.in", atoms_Angstrom, "espresso-in",
                     input_data=qe_input_data,
                     pseudopotentials=qe_pseudopotentials,
                     kpts=kpoints)

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

