"""Helper functions for several calculation tasks (such as integration)."""
from ase.units import kB
import mpmath as mp
import numpy as np
from scipy import integrate
import sys

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


def fermi_function(energy, fermi_energy, temperature,
                   suppress_overflow=False):
    r"""
    Calculate the Fermi function.

    The Fermi function reads

    .. math:: f(\epsilon) = 1/(1 + e^{\frac{\epsilon-\epsilon_F}{k_\mathrm{B} T}})

    Parameters
    ----------
    energy : float or numpy.array
        Energy for which the Fermi function is supposed to be calculated in
        energy_units.

    fermi_energy : float
        Fermi energy level in energy_units.

    temperature : float
        Temperature in K.

    suppress_overflow : bool
        If True, overflow that may occur in the exponent of the divisor of
        the Fermi function is suppressed. This is mathematically justifiable,
        but

    Returns
    -------
    fermi_val : float
        Value of the Fermi function.

    """
    exponent = (energy - fermi_energy) / (kB * temperature)

    if suppress_overflow:
        # Maximum exponent that will not result in inf when performing
        # np.exp() (for double values only).
        # Very large exponents result in very large values, and if we don't
        # check for overflow, infinity. Cutting this back to the maximum float
        # value (~ 1e308) is more then enough, since we divide by the exponent,
        # and therefore get zero (or machine precision within zero) either way.
        # Adding this check removes overflow warnings.
        # The overhead for inference should be minimal.
        # Since this function works both for arrays and scalars,
        # we have to check which maximum to use.

        if isinstance(energy, np.ndarray):
            max_exponent = np.log(np.finfo(exponent.dtype).max)
            exponent[exponent > max_exponent] = max_exponent
        else:
            exponent = min(exponent, np.log(np.finfo(exponent).max))

    return 1.0 / (1.0 + np.exp(exponent))


def entropy_multiplicator(energy, fermi_energy, temperature):
    r"""
    Calculate the multiplicator function for the entropy integral.

    The Entropy integral is

    .. math:: f(\epsilon)\log(f(\epsilon))+(1-f(\epsilon)\log(1-f(\epsilon))

    Parameters
    ----------
    energy : float or numpy.array
        Energy for which the Fermi function is supposed to be calculated in
        energy_units.

    fermi_energy : float
        Fermi energy level in energy_units.

    temperature : float
        Temperature in K.

    Returns
    -------
    multiplicator_val : float
        Value of the multiplicator function.
    """
    if len(np.shape(energy)) > 0:
        dim = np.shape(energy)[0]
        multiplicator = np.zeros(dim, dtype=np.float64)
        for i in range(0, np.shape(energy)[0]):
            fermi_val = fermi_function(energy[i], fermi_energy, temperature,
                                       suppress_overflow=True)
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
        fermi_val = fermi_function(energy, fermi_energy, temperature,
                                   suppress_overflow=True)
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


def get_beta(temperature):
    r"""
    Calculate :math:`\beta = {(k_\mathrm{B}T)}^{-1}`.

    Parameters
    ----------
    temperature : float
        Temperature in K

    Returns
    -------
    beta : float
        Thermodynamic beta.
    """
    return 1 / (kB * temperature)


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


def analytical_integration(D, I0, I1, fermi_energy, energy_grid, temperature):
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

    fermi_energy : float
        The fermi energy in eV.

    energy_grid : numpy.array
        Energy grid on which the integration should be performed.

    temperature : float
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

    # Calculate the weights.
    # It is not really possible to express this as a vector operation,
    # since mp.polylog (which is called in function_mappings) does not support
    # that.
    beta = 1 / (kB * temperature)
    for i in range(0, gridsize):
        # Some aliases for readibility
        ei = energy_grid_edges[i+1]
        ei_plus = energy_grid_edges[i+2]
        ei_minus = energy_grid_edges[i]

        # Calculate x
        x = beta*(ei - fermi_energy)
        x_plus = beta*(ei_plus - fermi_energy)
        x_minus = beta*(ei_minus - fermi_energy)

        # Calculate the I0 value
        i0 = function_mappings[I0](x, beta)
        i0_plus = function_mappings[I0](x_plus, beta)
        i0_minus = function_mappings[I0](x_minus, beta)

        # Calculate the I1 value
        i1 = function_mappings[I1](x, beta)
        i1_plus = function_mappings[I1](x_plus, beta)
        i1_minus = function_mappings[I1](x_minus, beta)

        weights_vector[i] = (i0_plus-i0) * (1 +
                                            ((ei - fermi_energy) / (ei_plus - ei))) \
                            + (i0-i0_minus) * (1 - ((ei - fermi_energy) / (ei - ei_minus))) - \
                            ((i1_plus-i1) / (ei_plus-ei)) + ((i1 - i1_minus)
                                                             / (ei - ei_minus))

    integral_value = np.dot(D, weights_vector)
    return integral_value


# Define Gaussian
def gaussians(grid, centers, sigma):
    r"""
    Calculate multiple gaussians on the same grid, but with different centers.

    Gaussian functions are used as approximations to the delta in the
    Brillouin zone integration. Note that this defines Gaussians without the
    factor of :math:`1/\sqrt{2}`. All the Gaussians will have the same sigmas.

    Parameters
    ----------
    grid : np.array
        Grid on which this Gaussian is defined.

    centers : np.array
        Array of centers for the Gaussians

    sigma : float
        Sigma value for the Gaussian.

    Returns
    -------
    multiple_gaussians : np.array
        multiple gaussians on the same grid, but with different centers.


    """
    multiple_gaussians = 1.0/np.sqrt(np.pi*sigma**2) * \
        np.exp(-1.0*((grid[np.newaxis] - centers[..., np.newaxis])/sigma)**2)

    return multiple_gaussians
