from .target_base import TargetBase
from .calculation_helpers import *


class DOS(TargetBase):
    """Density of states.
    Evaluation follow the outline and code provided in/by https://arxiv.org/abs/2010.04905.
    """
    def __init__(self, p):
        super(DOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize




    def get_band_energy(self, dos_data):
        return self.band_energy_from_dos(dos_data)

    @staticmethod
    def band_energy_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV,
                             temperature_K, integration_method):
        """
        Calculates the band energy from DOS data (directly, independent of the source of this DOS data.
        """

        # Calculate the energy levels and the Fermi function.
        energy_vals = np.linspace(emin, emax, nr_energy_levels)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculathe band energy.
        band_energy = integrate_values_on_grid(dos_data * (energy_vals * fermi_vals), energy_vals, axis=-1,
                                               method=integration_method)
        return band_energy