import scipy as sp
from mala.targets.calculation_helpers import *
import numpy as np
from mala.targets.ldos import LDOS
from mala.targets.density import Density
from mala.targets.dos import DOS
from mala.common.parameters import Parameters
from mala.common.parameters import printout
from data_repo_path import get_data_repo_path

# In order to test the integration capabilities of MALA we need a
# QuantumEspresso
# calculation containing the following:
#   1. Outfile from the run
#   2. LDOS cube files.
#   3. Density cube file.
#   4. A DOS file.
# In the check_integration_qe subfolder, input scripts can be found  for a
# simple test calculation (the results of this calculation have been used
# to develop this script). It uses an Al supercell with 36 atoms and a
# 6x6x6 k-grid.

data_path = get_data_repo_path()+"Al36/"
path_to_out = data_path+"Al.pw.scf.out"
path_to_ldos_qe = [data_path, "tmp.pp*Al_ldos.cube"]
path_to_dos_qe = [data_path, "Al.dos"]
path_to_dens_qe = [data_path, "Al_dens.cube"]
path_to_ldos_npy = data_path+"Al_ldos.npy"
path_to_dos_npy = data_path+"Al_dos.npy"
path_to_dens_npy = data_path+"Al_dens.npy"

# We can read from numpy arrays or directly from QE data.
# In the later case, numpy arrays will be saved for the subsqeuent run.
numpy_arrays = True


# Define the parameters used for the experiments.
test_parameters = Parameters()
test_parameters.targets.ldos_gridsize = 250
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10


#  Test our implementation of the analytical integration of [1]
def check_analytical_integration(accuracy):

    # Values used for the test.
    energies = np.array([6.83, 7.11])
    gridsize = 2
    e_fermi = 7.0
    temp = 298

    # Calculate the analytical values.
    beta = 1 / (kB * temp)
    x = beta * (energies - e_fermi)
    fi_0 = np.zeros(gridsize, dtype=np.float64)
    fi_1 = np.zeros(gridsize, dtype=np.float64)
    fi_2 = np.zeros(gridsize, dtype=np.float64)
    for i in range(0, gridsize):

        # Calculate beta and x
        fi_0[i] = get_f0_value(x[i], beta)
        fi_1[i] = get_f1_value(x[i], beta)
        fi_2[i] = get_f2_value(x[i], beta)
    aint_0 = fi_0[1] - fi_0[0]
    aint_1 = fi_1[1] - fi_1[0]
    aint_2 = fi_2[1] - fi_2[0]

    # Calculate the numerically approximated values.
    qint_0, abserr = sp.integrate.quad(
        lambda e: fermi_function_eV(e, e_fermi, temp),
        energies[0], energies[-1])
    qint_1, abserr = sp.integrate.quad(
        lambda e: (e - e_fermi) * fermi_function_eV(e, e_fermi, temp),
        energies[0], energies[-1])
    qint_2, abserr = sp.integrate.quad(
        lambda e: (e - e_fermi) ** 2 * fermi_function_eV(e, e_fermi, temp),
        energies[0], energies[-1])

    # Calculate the errors.
    error0 = np.abs(aint_0 - qint_0)
    error1 = np.abs(aint_1 - qint_1)
    error2 = np.abs(aint_2 - qint_2)

    # Analyze the errors.
    if error0 > accuracy or error1 > accuracy or error2 > accuracy:
        return False
    return True


# Integrate the QE density (Al.dens) over spatial grid. Does this yield the
# correct number of electrons?
def qe_dens_to_nr_of_electrons(accuracy):
    # Create a calculator.
    dens_calculator = Density(test_parameters)
    dens_calculator.read_additional_calculation_data("qe.out", path_to_out)

    # Read the input data.
    if numpy_arrays:
        density_dft = np.load(path_to_dens_npy)
    else:
        density_dft = dens_calculator.read_from_cube(path_to_dens_qe[1],
                                                     path_to_dens_qe[0])
        np.save(path_to_dens_npy, density_dft)

    # Calculate the quantities we want to compare.
    nr_mala = dens_calculator.get_number_of_electrons(density_dft)
    nr_dft = dens_calculator.number_of_electrons

    # Calculate relative error.
    rel_error = np.abs(nr_mala-nr_dft) / nr_dft
    printout("Relative error number of electrons: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < accuracy:
        return True
    else:
        return False


# Integrate QE LDOS over energy grid. Does this yield the correct density
# (when compared to Al.dens)?
def qe_ldos_to_density(accuracy):
    # Create a calculator.abs()
    ldos_calculator = LDOS(test_parameters)
    ldos_calculator.read_additional_calculation_data("qe.out", path_to_out)

    # Read the input data.
    density_dft = np.load(path_to_dens_npy)
    if numpy_arrays:
        ldos_dft = np.load(path_to_ldos_npy)
    else:
        ldos_dft = ldos_calculator.read_from_cube(path_to_ldos_qe[1],
                                                  path_to_ldos_qe[0])

        # LDOS is in 1/Ry. DOS is in 1/eV.
        ldos_dft = ldos_calculator.convert_units(ldos_dft, "1/Ry")
        np.save(path_to_ldos_npy, ldos_dft)

    # Calculate the quantities we want to compare.
    density_mala = ldos_calculator.get_density(ldos_dft)
    density_mala_sum = density_mala.sum()
    density_dft_sum = density_dft.sum()

    # Calculate relative error.
    rel_error = np.abs(density_mala_sum-density_dft_sum) / density_dft_sum
    printout("Relative error for sum of density: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < accuracy:
        return True
    else:
        return False


# Integrate the QE LDOS over spatial grid. Does this yield the corrected LDOS
# (when compared to Al.dos)?
def qe_ldos_to_dos(accuracy):
    # Create the necessary calculators.
    ldos_calculator = LDOS(test_parameters)
    ldos_calculator.read_additional_calculation_data("qe.out", path_to_out)
    dos_calculator = DOS(test_parameters)
    dos_calculator.read_additional_calculation_data("qe.out", path_to_out)

    # Read the input data.
    ldos_dft = np.load(path_to_ldos_npy)
    if numpy_arrays:
        dos_dft = np.load(path_to_dos_npy)
    else:
        # DOS is in 1/eV so no conversion necessary.
        dos_dft = dos_calculator.read_from_qe_dos_txt(path_to_dos_qe[1],
                                                      path_to_dos_qe[0])
        np.save(path_to_dos_npy, dos_dft)

    # Calculate the quantities we want to compare.
    dos_mala = ldos_calculator.get_density_of_states(ldos_dft)
    dos_mala_sum = dos_mala.sum()
    dos_dft_sum = dos_dft.sum()
    rel_error = np.abs(dos_mala_sum-dos_dft_sum) / dos_dft_sum
    printout("Relative error for sum of DOS: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < accuracy:
        return True
    else:
        return False


if __name__ == "__main__":

    # Run the tests.
    test1 = check_analytical_integration(0.0000001)
    printout("Check if analytical integration works in theory - success?:",
             test1)

    test1 = qe_dens_to_nr_of_electrons(0.0000001)
    printout("Integrate QE density over spatial grid and get correct number "
             "of electrons compared to QE - success?:", test1)

    test1 = qe_ldos_to_density(0.0000001)
    printout("Integrate QE LDOS over energy grid and get correct density "
             "compared to QE - succes?:", test1)

    test1 = qe_ldos_to_dos(0.0000001)
    printout("Integrate QE LDOS over spatial grid and get correct DOS "
             "compared to QE - success:", test1)
