import numpy as np
from ase.units import kB
from scipy import integrate


def integrate_values_on_grid(values, grid, axis=0, method="trapz"):
    """
    Integrates values on a given grid, using either the trapezoid or the simpson method.
    Input quantities:
        - values: Values to be integrated.
        - grid: Grid on which the value is supposed to be integrated
        - axis: Axis along which the integration is performed. Default=0,
        - method: Integration method to be used, can either be "trapz" for trapezoid or "simps" for Simpson method.
    """

    if method == "trapz":
        return integrate.trapz(values, grid, axis=axis)
    elif method == "simps":
        return integrate.simps(values, grid, axis=axis)
    else:
        raise Exception("Unknown integration method.")


def integrate_values_on_spacing(values, spacing, axis=0, method="trapz"):
    """
    Integrates values assuming a uniform grid with a provided spacing, using either the trapezoid or the simpson method.
    Input quantities:
        - values: Values to be integrated.
        - spacing: Spacing of the grid on which the integration is performed.
        - axis: Axis along which the integration is performed. Default=0,
        - method: Integration method to be used, can either be "trapz" for trapezoid or "simps" for Simpson method.
    """
    if method == "trapz":
        return integrate.trapz(values, dx=spacing, axis=axis)
    elif method == "simps":
        return integrate.simps(values, dx=spacing, axis=axis)
    else:
        raise Exception("Unknown integration method.")


def fermi_function(energy, fermi_energy, temperature_K, energy_units="eV"):
    """
    Calculates the fermi function
    f(E) = 1 / (1 + e^((E-E_F)/(k_B*T)))
    Input quantities:
        - energy_units: Currently supported: "eV"
        - energy: Energy for which the Fermi function is supposed to be calculated in energy_units
        - fermi_energy: Fermi energy level in energy_units
        - temperature_K: Temperature in K
    """
    if energy_units == "eV":
        return fermi_function_eV(energy, fermi_energy, temperature_K)


def fermi_function_eV(energy_ev, fermi_energy_ev, temperature_K):
    """
    Calculates the fermi function
    f(E) = 1 / (1 + e^((E-E_F)/(k_B*T)))
    Input quantities:
        - energy_ev: Energy for which the Fermi function is supposed to be calculated in eV
        - fermi_energy_ev: Fermi energy level in eV
        - temperature_K: Temperature in K
    """
    return 1.0 / (1.0 + np.exp((energy_ev - fermi_energy_ev) / (kB * temperature_K)))
