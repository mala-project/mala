from ase.units import Rydberg, Bohr, kB
import ase.io
import numpy as np
from fesl.common.parameters import Parameters, ParametersTargets


class TargetBase:
    """Base class for a target quantity parser. Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file format."""
    def __init__(self, p):
        if isinstance(p, Parameters):
            self.parameters = p.targets
        elif isinstance(p, ParametersTargets):
            self.parameters = p
        else:
            raise Exception("Wrong type of parameters for Targets class.")
        self.target_length = 0
        self.fermi_energy_eV = None
        self.temperature_K = None
        self.grid_spacing_Bohr = None
        self.number_of_electrons = None
        self.band_energy_dft_calculation = None

    def read_from_cube(self):
        raise Exception("No function defined to read this quantity from a .cube file.")

    def read_from_qe_dos_txt(self):
        raise Exception("No function defined to read this quantity from a qe.dos.txt file")

    def get_density(self):
        raise Exception("No function to calculate or provide the density has been implemented for this target type.")

    def get_density_of_states(self):
        raise Exception("No function to calculate or provide the density of states (DOS) has been implemented for this target type.")

    def get_band_energy(self):
        raise Exception("No function to calculate or provide the band energy has been implemented for this target type.")

    def get_number_of_electrons(self):
        raise Exception("No function to calculate or provide the number of electrons has been implemented for this target type.")

    def read_additional_calculation_data(self, data_type, path_to_file=""):
        """
        Reads in additional input about a calculation. This is e.g. necessary when we operate with preprocessed
        data for the training itself but want to take into account other physical quantities (such as the fermi energy
        or the electronic temperature) for post processing.
        Inputs:
            - data_type: Type of data or file that is used. Currently supports qe.out for Quantum Espresso outfiles.
            - path_to_file: Path to the file that is used.
        """
        if data_type == "qe.out":
            atoms = ase.io.read(path_to_file, format="espresso-out")
            vol = atoms.get_volume()
            cell_volume = vol / (200 * 200 * 200 * Bohr ** 3)
            self.grid_spacing_Bohr = cell_volume**(1/3)
            self.fermi_energy_eV = atoms.get_calculator().get_fermi_level()

            with open(path_to_file) as out:
                for line in out:
                    if "number of electrons       =" in line:
                        self.number_of_electrons = np.float64(line.split('=')[1])
                    if "Fermi-Dirac smearing, width (Ry)=" in line:
                        self.temperature_K = np.float64(line.split('=')[2]) * Rydberg / kB
                    if "xc contribution" in line:
                        xc_contribution = float((line.split('=')[1]).split('Ry')[0])
                        break
                    if "one-electron contribution" in line:
                        one_electron_contribution = float((line.split('=')[1]).split('Ry')[0])
                    if "hartree contribution" in line:
                        hartree_contribution = float((line.split('=')[1]).split('Ry')[0])
            band_energy_Ry = one_electron_contribution + xc_contribution + hartree_contribution
            self.band_energy_dft_calculation = band_energy_Ry*Rydberg
        else:
            raise Exception("Unsupported auxiliary file type.")

    @staticmethod
    def convert_units(array, in_units="eV"):
        raise Exception("No unit conversion method implemented for this target type.")

    @staticmethod
    def backconvert_units(array, out_units):
        raise Exception("No unit back conversion method implemented for this target type.")