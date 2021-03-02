"""Helper functions for several calculation tasks (such as integration)."""
import numpy as np
from ase.units import kB
from scipy import integrate, interpolate
import mpmath as mp


def integrate_values_on_spacing(values, spacing, method, axis=0):
    """
    Integrate values assuming a uniform grid with a provided spacing.

    Different integration methods are available.

    Parameters
    ----------
    values : numpy.array
        Values to be integrated.
    spacing : int
        Spacing of the grid on which the integration is performed.
    method : string
        Integration method to be used. Currently supported:

            - "trapz" for trapezoid method
            - "simps" for Simpson method.
    axis : int
        Axis along which the integration is performed.

    Returns
    -------
    integral_values : float
        The value of the integral.
    """
    if method == "trapz":
        return integrate.trapz(values, dx=spacing, axis=axis)
    elif method == "simps":
        return integrate.simps(values, dx=spacing, axis=axis)
    else:
        raise Exception("Unknown integration method.")


def fermi_function(energy, fermi_energy, temperature_K, energy_units="eV"):
    """
    Calculate the Fermi function f(E) = 1 / (1 + e^((E-E_F)/(k_B*T))).

    Parameters
    ----------
    energy : float
        Energy for which the Fermi function is supposed to be calculated in
        energy_units.
    fermi_energy : float
        Fermi energy level in energy_units.
    temperature_K : float
        Temperature in K.
    energy_units : string
        Currently supported:

            - eV
    Returns
    -------
    fermi_val : float
        Value of the Fermi function.

    """
    if energy_units == "eV":
        return fermi_function_eV(energy, fermi_energy, temperature_K)


def entropy_multiplicator(energy, fermi_energy, temperature_K,
                          energy_units="eV"):
    """
    Calculate the multiplicator function for the entropy integral.

    Entropy integral is f(E)*log(f(E))+(1-f(E)*log(1-f(E))

    Parameters
    ----------
    energy : float
        Energy for which the Fermi function is supposed to be calculated in
        energy_units.
    fermi_energy : float
        Fermi energy level in energy_units.
    temperature_K : float
        Temperature in K.
    energy_units : string
        Currently supported:

            - eV

    Returns
    -------
    multiplicator_val : float
        Value of the multiplicator function.
    """
    if len(np.shape(energy)) > 0:
        dim = np.shape(energy)[0]
        multiplicator = np.zeros(dim, dtype=np.float64)
        for i in range(0, np.shape(energy)[0]):
            fermi_val = fermi_function(energy[i], fermi_energy, temperature_K,
                                       energy_units=energy_units)
            if fermi_val == 1.0:
                secondterm = 0.0
            else:
                secondterm = (1 - fermi_val) * np.log(1 - fermi_val)
            if fermi_val == 0.0:
                firsterm = 0.0
            else:
                firsterm = fermi_val * np.log(fermi_val)
            multiplicator[i] = firsterm + secondterm
    else:
        fermi_val = fermi_function(energy, fermi_energy, temperature_K,
                                   energy_units=energy_units)
        if fermi_val == 1.0:
            secondterm = 0.0
        else:
            secondterm = (1 - fermi_val) * np.log(1 - fermi_val)
        if fermi_val == 0.0:
            firsterm = 0.0
        else:
            firsterm = fermi_val * np.log(fermi_val)
        multiplicator = firsterm + secondterm

    return multiplicator


def fermi_function_eV(energy_ev, fermi_energy_ev, temperature_K):
    """
    Calculate the Fermi function f(E) = 1 / (1 + e^((E-E_F)/(k_B*T))).

    Parameters
    ----------
    energy_ev : float
        Energy in eV.
    fermi_energy_ev : float
        Fermi energy level in eV.
    temperature_K : float
        Temperature in K.

    Returns
    -------
    fermi_val : float
        Value of the Fermi function.
    """
    return 1.0 / (1.0 + np.exp((energy_ev - fermi_energy_ev) /
                               (kB * temperature_K)))


def get_beta(temperature_K):
    """
    Calculate beta = 1 / (k_B * T).

    Parameters
    ----------
    temperature_K : float
        Temperature in K

    Returns
    -------
    beta : float
        Thermodynamic beta.
    """
    return 1 / (kB * temperature_K)


def get_f0_value(x, beta):
    """
    Get the F0 value for the analytic integration formula.

    Parameters
    ----------
    x : float
        x value for function.

    beta : float
        Thermodynamic beta.

    Returns
    -------
    function_value : float
        F0 value.
    """
    results = (x+mp.polylog(1, -1.0*mp.exp(x)))/beta
    return results


def get_f1_value(x, beta):
    """
    Get the F1 value for the analytic integration formula.

    Parameters
    ----------
    x : float
        x value for function.

    beta : float
        Thermodynamic beta.

    Returns
    -------
    function_value : float
        F1 value.
    """
    results = ((x*x)/2+x*mp.polylog(1, -1.0*mp.exp(x))
               - mp.polylog(2, -1.0*mp.exp(x))) / (beta*beta)
    return results


def get_f2_value(x, beta):
    """
    Get the F2 value for the analytic integration formula.

    Parameters
    ----------
    x : float
        x value for function.

    beta : float
        Thermodynamic beta.

    Returns
    -------
    function_value : float
        F2 value.
    """
    results = ((x*x*x)/3+x*x*mp.polylog(1, -1.0*mp.exp(x)) -
               2*x*mp.polylog(2, -1.0*mp.exp(x)) +
               2*mp.polylog(3, -1.0*mp.exp(x))) / (beta*beta*beta)
    return results


def get_s0_value(x, beta):
    """
    Get the S0 value for the analytic integration formula.

    Parameters
    ----------
    x : float
        x value for function.

    beta : float
        Thermodynamic beta.

    Returns
    -------
    function_value : float
        S0 value.
    """
    results = (-1.0*x*mp.polylog(1, -1.0*mp.exp(x)) +
               2.0*mp.polylog(2, -1.0*mp.exp(x))) / (beta*beta)
    return results


def get_s1_value(x, beta):
    """
    Get the S1 value for the analytic integration formula.

    Parameters
    ----------
    x : float
        x value for function.

    beta : float
        Thermodynamic beta.

    Returns
    -------
    function_value : float
        S1 value.
    """
    results = (-1.0*x*x*mp.polylog(1, -1.0*mp.exp(x)) +
               3*x*mp.polylog(2, -1.0*mp.exp(x)) -
               3*mp.polylog(3, -1.0*mp.exp(x))) / (beta*beta*beta)
    return results


def analytical_integration(D, I0, I1, fermi_energy_ev, energy_grid,
                           temperature_k):
    """
    Perform analytical integration following the outline given by [1].

    More specifically, this function solves Eq. 32, by constructing the
    weights itself.

    Parameters
    ----------
    D : numpy.array or float
        Either LDOS or DOS data.

    I0 : string
        I_0 from Eq. 33. The user only needs to provide which function should
        be used. Arguments are:

            - F0
            - F1
            - F2
            - S0
            - S1

    I1 : string
        I_1 from Eq. 33. The user only needs to provide which function should
        be used. Arguments are:

            - F0
            - F1
            - F2
            - S0
            - S1

    fermi_energy_ev : float
        The fermi energy in eV.

    energy_grid : numpy.array
        Energy grid on which the integration should be performed.

    temperature_k : float
        Temperature in K.

    Returns
    -------
    integration_value : numpy.array or float
        Value of the integral.
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
    if I0 not in list(function_mappings.keys()) or I1 not in\
            list(function_mappings.keys()):
        raise Exception("Could not calculate analytical intergal, "
                        "wrong choice of auxiliary functions.")

    # Construct the weight vector.
    weights_vector = np.zeros(energy_grid.shape, dtype=np.float64)
    gridsize = energy_grid.shape[0]
    energy_grid_edges = np.zeros(energy_grid.shape[0]+2, dtype=np.float64)
    energy_grid_edges[1:-1] = energy_grid
    spacing = (energy_grid[1]-energy_grid[0])
    energy_grid_edges[0] = energy_grid[0] - spacing
    energy_grid_edges[-1] = energy_grid[-1] + spacing

    if len(D.shape) > 1:
        real_space_grid = D.shape[0]
        integral_value = np.zeros(real_space_grid, dtype=np.float64)
    else:
        real_space_grid = 1
        integral_value = 0

    # Calculate the weights.
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

        weights_vector[i] = (i0_plus-i0)*(1 +
                                          ((ei-fermi_energy_ev)/(ei_plus-ei)))\
            + (i0-i0_minus)*(1-((ei-fermi_energy_ev)/(ei-ei_minus))) - \
                            ((i1_plus-i1) / (ei_plus-ei)) + ((i1 - i1_minus)
                                                             / (ei - ei_minus))
        if real_space_grid == 1:
            integral_value += weights_vector[i] * D[i]
        else:
            for j in range(0, real_space_grid):
                integral_value[j] += weights_vector[i] * D[j, i]
    return integral_value

