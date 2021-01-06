import numpy as np
from ase.units import kB
from scipy import integrate
import mpmath as mp

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


def get_f0_value(x, beta):
    results = (x+mp.polylog(1, -1.0*mp.exp(x)))/beta
    return results


def get_f1_value(x, beta):
    results = ((x*x)/2+x*mp.polylog(1, -1.0*mp.exp(x)) - mp.polylog(2, -1.0*mp.exp(x))) / (beta*beta)
    return results


def get_f2_value(x, beta):
    results = ((x*x*x)/3+x*x*mp.polylog(1, -1.0*mp.exp(x)) - 2*x*mp.polylog(2, -1.0*mp.exp(x)) + 2*mp.polylog(3, -1.0*mp.exp(x))) / (beta*beta*beta)
    return results


def get_s0_value(x, beta):
    results = (-1.0*x*mp.polylog(1, -1.0*mp.exp(x)) - 2.0*mp.polylog(2, -1.0*mp.exp(x))) / (beta*beta)
    return results


def get_s1_value(x, beta):
    results = (-1.0*x*x*mp.polylog(1, -1.0*mp.exp(x)) + 3*x*mp.polylog(2, -1.0*mp.exp(x)) - 3*mp.polylog(3, -1.0*mp.exp(x))) / (beta*beta*beta)
    return results


def analytical_integration(D, I0, I1, fermi_energy_ev, energy_grid, temperature_k):
    """
    Analytical integration following the outline given by [1].
    More specifically, this function solves Eq. 32, by constructing the weights itself.
    Input quantities:
        - D: Either DOS or LDOS data.
        - I0: I_0 from Eq. 33. The user only needs to provide which function should be used.
        - I1: I_1 from Eq. 33. The user only needs to provide which function should be used.
        - Arguments for I0 and I1 are:
            "F0", "F1", "F2", "S0", "S1"
        - fermi_energy_eV: The fermi energy in eV
        - energy_grid: (L)DOS energy grid.
        - emax: Upper end of the (L)DOS energy grid.
        - nr_energy_levels: Number of points on the (L)DOS energy grid.
    """

    # Mappings for the functions further down.
    function_mappings = {
        "F0": get_f0_value,
        "F1": get_f1_value,
        "F2": get_f2_value,
        "S0": get_s0_value,
        "S1": get_s1_value,
    }

    # Check if everything makes sense.
    if I0 not in list(function_mappings.keys()) or I1 not in list(function_mappings.keys()):
        raise Exception("Could not calculate analytical intergal, wrong choice of auxiliary functions.")

    # Construct the weight vector.
    weights_vector = np.zeros(energy_grid.shape, dtype=np.float)
    gridsize = energy_grid.shape[0]
    energy_grid_edges = np.zeros(energy_grid.shape[0]+2, dtype=np.float)
    energy_grid_edges[1:-1] = energy_grid
    spacing = (energy_grid[1]-energy_grid[0])
    energy_grid_edges[0] = energy_grid[0] - spacing
    energy_grid_edges[-1] = energy_grid[-1] + spacing

    # Calculate the weights.
    integral_value = 0
    for i in range(0, gridsize):
        # Calculate beta and x
        beta = 1 / (kB * temperature_k)
        x = beta*(energy_grid_edges[i]-fermi_energy_ev)
        x_plus = beta*(energy_grid_edges[i+1]-fermi_energy_ev)
        x_minus = beta*(energy_grid_edges[i-1]-fermi_energy_ev)

        # Calculate the I0 value
        i0 = function_mappings[I0](x, beta)
        i0_plus = function_mappings[I0](x_plus, beta)
        i0_minus = function_mappings[I0](x_minus, beta)

        # Calculate the I1 value
        i1 = function_mappings[I1](x, beta)
        i1_plus = function_mappings[I1](x_plus, beta)
        i1_minus = function_mappings[I1](x_minus, beta)

        # Some aliases for readibility
        ei = energy_grid_edges[i]
        ei_plus = energy_grid_edges[i+1]
        ei_minus = energy_grid_edges[i-1]

        weights_vector[i] = (i0_plus-i0)*(1 + ((ei-fermi_energy_ev)/(ei_plus-ei))) \
                            + (i0-i0_minus)*(1-((ei-fermi_energy_ev)/(ei-ei_minus))) \
                            - ((i1_plus-i1) / (ei_plus-ei)) +((i1 - i1_minus) /(ei - ei_minus))

        integral_value += weights_vector[i] * D[i]

    return integral_value

