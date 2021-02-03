from .target_base import TargetBase
from .calculation_helpers import *
from scipy import integrate, interpolate
from scipy.optimize import toms748
from ase.units import Rydberg
from fesl.common.parameters import printout


class DOS(TargetBase):
    """Density of states.
    Evaluation follow the outline and code provided in/by https://arxiv.org/abs/2010.04905.
    """
    def __init__(self, p):
        super(DOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Converts the units of an LDOS array. Can be used e.g. for post processing purposes.
        This function always returns the LDOS in 1/eV.
        """
        if in_units == "1/eV":
            return array
        elif in_units == "1/Ry":
            return array / Rydberg
        else:
            printout(in_units)
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Converts the units of an LDOS array back from internal use to outside use.
        Can be used e.g. for post processing purposes.
        This function always accepts the LDOS in 1/eV and outputs them in the desired unit.
        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            printout(out_units)
            raise Exception("Unsupported unit for LDOS.")

    def read_from_qe_dos_txt(self, file_name, directory):
        """
        Reads the DOS from a Quantum Espresso generated file.
        These files do not have a specified file ending, so I will call them
        qe.dos.txt here.
        """
        # Create the desired/specified energy grid. We will use this to check whether we have a correct file.

        energy_grid = np.arange(self.parameters.ldos_gridoffset_ev, self.parameters.ldos_gridoffset_ev+self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev,self.parameters.ldos_gridspacing_ev)
        return_dos_values = []

        # Open the file, then iterate through its contents.
        with open(directory+file_name, 'r') as infile:
            lines = infile.readlines()
            i = 0

            for dos_line in lines:
                # The first column contains the energy value.
                if "#" not in dos_line and i < self.parameters.ldos_gridsize:
                    eval = float(dos_line.split()[0])
                    dosval = float(dos_line.split()[1])
                    if np.abs(eval-energy_grid[i]) < self.parameters.ldos_gridspacing_ev*0.98:
                        return_dos_values.append(dosval)
                        i += 1

        return np.array(return_dos_values)

    def get_energy_grid(self):
        return np.arange(self.parameters.ldos_gridoffset_ev, self.parameters.ldos_gridoffset_ev+self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev,self.parameters.ldos_gridspacing_ev)


    def get_band_energy(self, dos_data, fermi_energy_eV=None, temperature_K=None, integration_method="analytical", shift_energy_grid=True):
        """
        Calculates the band energy, from given DOS data.
        Input variables:
            - dos_data - This method can only be called if the DOS data is presented in the form:
                    [energygrid]
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.__band_energy_from_dos(dos_data, emin, emax, self.parameters.ldos_gridspacing_ev, fermi_energy_eV, temperature_K,
                                           integration_method, shift_energy_grid)

    def get_number_of_electrons(self, dos_data, fermi_energy_eV=None, temperature_K=None, integration_method="analytical", shift_energy_grid=True):
        """
        Calculates the number of electrons, from given DOS data.
        Input variables:
            - dos_data - This method can only be called if the DOS data is presented in the form:
                    [energygrid]
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.__number_of_electrons_from_dos(dos_data, emin, emax, self.parameters.ldos_gridspacing_ev, fermi_energy_eV,
                                                   temperature_K, integration_method, shift_energy_grid)

    def get_entropy_contribution(self, dos_data, fermi_energy_eV=None, temperature_K=None, integration_method="analytical", shift_energy_grid=True):
        """
        Calculates the contribution of the entropy to the total energy, from given DOS data.
        Input variables:
            - dos_data - This method can only be called if the DOS data is presented in the form:
                    [energygrid]
        """
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.__entropy_contribution_from_dos(dos_data, emin, emax, self.parameters.ldos_gridspacing_ev, fermi_energy_eV,
                                                   temperature_K, integration_method, shift_energy_grid)



    def get_self_consistent_fermi_energy_ev(self, dos_data, temperature_K=None, integration_method="analytical", shift_energy_grid=True):
        """
        Calculates the "self-consistent Fermi energy", i.e. the fermi energy for which integrating over the DOS
        gives the EXACT number of atoms.
        """

        # Parse the parameters.
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize * self.parameters.ldos_gridspacing_ev + self.parameters.ldos_gridoffset_ev

        fermi_energy_sc = toms748(lambda fermi_sc: (self.__number_of_electrons_from_dos(dos_data, emin, emax,
                                                        self.parameters.ldos_gridspacing_ev, fermi_sc,
                                                        temperature_K, integration_method, shift_energy_grid)
                                                        - self.number_of_electrons), a=emin, b=emax)
        return fermi_energy_sc

    def get_density_of_states(self, dos_data):
        """If the target is the DOS, recovering the DOS is rather trivial."""
        return dos_data


    @classmethod
    def from_ldos(cls, ldos_object):
        """
        Create a DOS calculator from an LDOS object. This is often necessary as a lot of post processing is actually
        done on the DOS rather than the LDOS.
        """
        return_dos_object = DOS(ldos_object.parameters)
        return_dos_object.fermi_energy_eV = ldos_object.fermi_energy_eV
        return_dos_object.temperature_K = ldos_object.temperature_K
        return_dos_object.grid_spacing_Bohr = ldos_object.grid_spacing_Bohr
        return_dos_object.number_of_electrons = ldos_object.number_of_electrons
        return_dos_object.band_energy_dft_calculation = ldos_object.band_energy_dft_calculation
        return_dos_object.atoms = ldos_object.atoms
        return_dos_object.qe_input_data = ldos_object.qe_input_data
        return_dos_object.qe_pseudopotentials = ldos_object.qe_pseudopotentials

        return return_dos_object



    @staticmethod
    def __number_of_electrons_from_dos(dos_data, emin, emax, energy_grid_spacing, fermi_energy_eV,
                                       temperature_K, integration_method, shift_energy_grid):
        """
        Calculates the number of electrons from DOS data (directly, independent of the source of this DOS data).
        I don't fully understand why shift_energy_grid yet. But it definitely is.
        """
        # Calculate the energy levels and the Fermi function.

        # energy_vals = np.linspace(emin, emax, nr_energy_levels)
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculate the number of electrons.
        if integration_method == "trapz":
            number_of_electrons = integrate.trapz(dos_data * fermi_vals, energy_vals, axis=-1)
        elif integration_method == "simps":
            number_of_electrons = integrate.simps(dos_data * fermi_vals, energy_vals, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            number_of_electrons, abserr = integrate.quad(
                lambda e: dos_pointer(e) * fermi_function(e, fermi_energy_eV, temperature_K),
                energy_vals[0], energy_vals[-1], limit=500, points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1", fermi_energy_eV, energy_vals, temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return number_of_electrons

    @staticmethod
    def __band_energy_from_dos(dos_data, emin, emax, energy_grid_spacing, fermi_energy_eV,
                               temperature_K, integration_method, shift_energy_grid):
        """
        Calculates the band energy from DOS data (directly, independent of the source of this DOS data).
        """

        # Calculate the energy levels and the Fermi function.
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculate the band energy.
        if integration_method == "trapz":
            band_energy = integrate.trapz(dos_data * (energy_vals * fermi_vals), energy_vals, axis=-1)
        elif integration_method == "simps":
            band_energy = integrate.simps(dos_data * (energy_vals * fermi_vals), energy_vals, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            band_energy, abserr = integrate.quad(
                lambda e: dos_pointer(e) * e * fermi_function(e, fermi_energy_eV, temperature_K),
                energy_vals[0], energy_vals[-1], limit=500, points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1", fermi_energy_eV, energy_vals, temperature_K)
            band_energy_minus_uN = analytical_integration(dos_data, "F1", "F2", fermi_energy_eV, energy_vals, temperature_K)
            band_energy = band_energy_minus_uN+fermi_energy_eV*number_of_electrons
        else:
            raise Exception("Unknown integration method.")

        return band_energy

    @staticmethod
    def __entropy_contribution_from_dos(dos_data, emin, emax, energy_grid_spacing, fermi_energy_eV,
                               temperature_K, integration_method, shift_energy_grid):
        """
        Calculates the entropy contribution to the total energy from DOS data (directly,
        independent of the source of this DOS data).
        More specifically, this gives -\beta^-1*S_S
        """

        # Calculate the energy levels and the Fermi function.
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculate the entropy contribution to the energy.
        if integration_method == "trapz":
            multiplicator = entropy_multiplicator(energy_vals, fermi_energy_eV, temperature_K)
            entropy_contribution = integrate.trapz(dos_data * multiplicator, energy_vals, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "simps":
            multiplicator = entropy_multiplicator(energy_vals, fermi_energy_eV, temperature_K)
            entropy_contribution = integrate.simps(dos_data * multiplicator, energy_vals, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            entropy_contribution, abserr = integrate.quad(
                lambda e: dos_pointer(e) * entropy_multiplicator(e, fermi_energy_eV, temperature_K),
                energy_vals[0], energy_vals[-1], limit=500, points=fermi_energy_eV)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "analytical":
            entropy_contribution = analytical_integration(dos_data, "S0", "S1", fermi_energy_eV, energy_vals, temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return entropy_contribution
