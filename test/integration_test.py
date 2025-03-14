import os

from mala import LDOS, Density, DOS, Parameters, printout
from mala.targets.calculation_helpers import *
import numpy as np
import scipy as sp
import pytest

from mala.datahandling.data_repo import data_path

# In order to test the integration capabilities of MALA we need a
# QuantumEspresso
# calculation containing the following:
#   1. Outfile from the run
#   2. LDOS cube files.
#   3. Density cube file.
#   4. A DOS file.
# Scripts to reproduce the data files used in this test script can be found
# in the data repo.

path_to_out = os.path.join(data_path, "Be_snapshot0.out")
path_to_ldos_npy = os.path.join(data_path, "Be_snapshot0.out.npy")
path_to_dos_npy = os.path.join(data_path, "Be_snapshot0.dos.npy")
path_to_dens_npy = os.path.join(data_path, "Be_snapshot0.dens.npy")

# We can read from numpy arrays or directly from QE data.
# In the later case, numpy arrays will be saved for the subsqeuent run.
numpy_arrays = True

# Define the parameters used for the experiments.
test_parameters = Parameters()
test_parameters.targets.ldos_gridsize = 11
test_parameters.targets.ldos_gridspacing_ev = 2.5
test_parameters.targets.ldos_gridoffset_ev = -5

# Define the accuracy used in the tests.
accuracy = 1e-6
accuracy_dos = 1e-4
accuracy_ldos = 0.2


class TestMALAIntegration:
    """
    Test class for MALA's integation routines.

    Tests different integrations that would normally be performed by code.
    """

    def test_analytical_integration(self):
        """
        Test whether the analytical integration works in principle.
        """
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
            lambda e: fermi_function(e, e_fermi, temp, suppress_overflow=True),
            energies[0],
            energies[-1],
        )
        qint_1, abserr = sp.integrate.quad(
            lambda e: (e - e_fermi)
            * fermi_function(e, e_fermi, temp, suppress_overflow=True),
            energies[0],
            energies[-1],
        )
        qint_2, abserr = sp.integrate.quad(
            lambda e: (e - e_fermi) ** 2
            * fermi_function(e, e_fermi, temp, suppress_overflow=True),
            energies[0],
            energies[-1],
        )

        # Calculate the errors.
        error0 = np.abs(aint_0 - qint_0)
        error1 = np.abs(aint_1 - qint_1)
        error2 = np.abs(aint_2 - qint_2)

        # Analyze the errors.
        assert np.isclose(error0, 0, atol=accuracy)
        assert np.isclose(error1, 0, atol=accuracy)
        assert np.isclose(error2, 0, atol=accuracy)

    def test_qe_dens_to_nr_of_electrons(self):
        """
        Test integration of density on real space grid.

        The integral of the density of the real space grid should yield the
        number of electrons.
        """
        # Create a calculator.
        dens_calculator = Density(test_parameters)
        dens_calculator.read_additional_calculation_data(
            path_to_out, "espresso-out"
        )

        # Read the input data.
        density_dft = np.load(path_to_dens_npy)

        # Calculate the quantities we want to compare.
        nr_mala = dens_calculator.get_number_of_electrons(density_dft)
        nr_dft = dens_calculator.number_of_electrons_exact

        # Calculate relative error.
        rel_error = np.abs(nr_mala - nr_dft) / nr_dft
        printout(
            "Relative error number of electrons: ", rel_error, min_verbosity=0
        )

        # Check against the constraints we put upon ourselves.
        assert np.isclose(rel_error, 0, atol=accuracy)

    @pytest.mark.skipif(
        os.path.isfile(path_to_ldos_npy) is False,
        reason="No LDOS file in data repo found.",
    )
    def test_qe_ldos_to_density(self):
        """
        Test integration of local density of states on energy grid.

        The integral of the LDOS over energy grid should yield the density.
        """
        # Create a calculator.abs()
        ldos_calculator = LDOS(test_parameters)
        ldos_calculator.read_additional_calculation_data(
            path_to_out, "espresso-out"
        )
        dens_calculator = Density.from_ldos_calculator(ldos_calculator)

        # Read the input data.
        density_dft = np.load(path_to_dens_npy)
        ldos_dft = np.load(path_to_ldos_npy)

        # Calculate the quantities we want to compare.
        self_consistent_fermi_energy = (
            ldos_calculator.get_self_consistent_fermi_energy(ldos_dft)
        )
        density_mala = ldos_calculator.get_density(
            ldos_dft, fermi_energy=self_consistent_fermi_energy
        )
        density_mala_sum = density_mala.sum()
        density_dft_sum = density_dft.sum()

        # Calculate relative error.
        rel_error = (
            np.abs(density_mala_sum - density_dft_sum) / density_dft_sum
        )
        printout(
            "Relative error for sum of density: ", rel_error, min_verbosity=0
        )

        # Check against the constraints we put upon ourselves.
        assert np.isclose(rel_error, 0, atol=accuracy)

    @pytest.mark.skipif(
        os.path.isfile(path_to_ldos_npy) is False,
        reason="No LDOS file in data repo found.",
    )
    def test_qe_ldos_to_dos(self):
        """
        Test integration of local density of states on real space grid.

        The integral of the LDOS over real space grid should yield the DOS.
        """
        ldos_calculator = LDOS(test_parameters)
        ldos_calculator.read_additional_calculation_data(
            path_to_out, "espresso-out"
        )
        dos_calculator = DOS(test_parameters)
        dos_calculator.read_additional_calculation_data(
            path_to_out, "espresso-out"
        )

        # Read the input data.
        ldos_dft = np.load(path_to_ldos_npy)
        dos_dft = np.load(path_to_dos_npy)

        # Calculate the quantities we want to compare.
        dos_mala = ldos_calculator.get_density_of_states(ldos_dft)
        dos_mala_sum = dos_mala.sum()
        dos_dft_sum = dos_dft.sum()
        rel_error = np.abs(dos_mala_sum - dos_dft_sum) / dos_dft_sum
        printout("Relative error for sum of DOS: ", rel_error, min_verbosity=0)

        # Check against the constraints we put upon ourselves.
        assert np.isclose(rel_error, 0, atol=accuracy_ldos)

    def test_pwevaldos_vs_ppdos(self):
        """Check pp.x DOS vs. pw.x DOS (from eigenvalues in outfile)."""
        dos_calculator = DOS(test_parameters)
        dos_calculator.read_additional_calculation_data(
            path_to_out, "espresso-out"
        )

        dos_from_pp = np.load(path_to_dos_npy)

        # Calculate the quantities we want to compare.
        dos_calculator.read_from_qe_out()
        dos_from_dft = dos_calculator.density_of_states
        dos_pp_sum = dos_from_pp.sum()
        dos_dft_sum = dos_from_dft.sum()
        rel_error = np.abs(dos_dft_sum - dos_pp_sum) / dos_pp_sum

        assert np.isclose(rel_error, 0, atol=accuracy_dos)
