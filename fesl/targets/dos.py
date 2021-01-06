from .target_base import TargetBase
from .calculation_helpers import *


class DOS(TargetBase):
    """Density of states.
    Evaluation follow the outline and code provided in/by https://arxiv.org/abs/2010.04905.
    """
    def __init__(self, p):
        super(DOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

    def get_band_energy(self, dos_data, fermi_energy_eV, temperature_K, integration_method="simps"):
        """
        Calculates the band energy, from given DOS data.
        Input variables:
            - dos_data - This method can only be called if the DOS data is presented in the form:
                    [energygrid]
        """

        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        nr_energy_levels = self.parameters.ldos_gridsize
        return self.band_energy_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV, temperature_K,
                                         integration_method)

    def get_number_of_electrons(self, dos_data, fermi_energy_eV, temperature_K, integration_method="simps"):
        """
        Calculates the number of electrons, from given DOS data.
        Input variables:
            - dos_data - This method can only be called if the DOS data is presented in the form:
                    [energygrid]
        """
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        nr_energy_levels = self.parameters.ldos_gridsize
        return self.number_of_electrons_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV,
                             temperature_K, integration_method)

    @staticmethod
    def number_of_electrons_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV,
                             temperature_K, integration_method="simps"):
        """
        Calculates the number of electrons from DOS data (directly, independent of the source of this DOS data).
        """
        # Calculate the energy levels and the Fermi function.
        energy_vals = np.linspace(emin, emax, nr_energy_levels)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        if integration_method != "analytical":
            # regular integration:
            number_of_electrons = integrate_values_on_grid(dos_data * fermi_vals, energy_vals, axis=-1,
                                               method=integration_method)
        else:
            # using regular integration.
            number_of_electrons = analytical_integration(dos_data, "F0", "F1", fermi_energy_eV, energy_vals, temperature_K)

        return number_of_electrons

    @staticmethod
    def band_energy_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV,
                             temperature_K, integration_method="simps"):
        """
        Calculates the band energy from DOS data (directly, independent of the source of this DOS data).
        """

        # Calculate the energy levels and the Fermi function.
        energy_vals = np.linspace(emin, emax, nr_energy_levels)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculathe band energy.
        band_energy = integrate_values_on_grid(dos_data * (energy_vals * fermi_vals), energy_vals, axis=-1,
                                               method=integration_method)
        return band_energy
