import fesl
from fesl import printout
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

"""
ex09_get_total_energy.py: Shows how FESL can be used to get the total energy of a system. For this, QuantumEspresso
is interfaced. Make sure you have Quantum Espresso and the total_energy python module installed before running this
example (see install/INSTALL_TE_QE.md). Also, since we are using QuantumEspresso, we cannot simply use downscaled
data from a bigger calculation. Instead, we rely on preprocessed data from a smaller calculation to analyze these results. 
"""

def run_example09(accuracy = 50):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = fesl.Parameters()
    test_parameters.targets.ldos_gridsize = 250
    test_parameters.targets.ldos_gridspacing_ev = 0.1
    test_parameters.targets.ldos_gridoffset_ev = -10.0

    ####################
    # CALCULATORS
    # Set up a calculator and use it.
    ####################
    ldos_calculator = fesl.LDOS(test_parameters)
    ldos_calculator.read_additional_calculation_data("qe.out", data_path+"Al.pw.scf.out")

    # Note: This repo only contains ONE pseudopotential. If you work with different elements, you will have to get
    # additional pseudopotentials.
    ldos_calculator.set_pseudopotential_path(data_path)
    dos_data = np.load(data_path+"Al_dos.npy")
    dens_data = np.load(data_path+"Al_dens.npy")

    # To get best results, it is advised to get the self-consistent fermi energy.
    dos_calculator = fesl.DOS.from_ldos(ldos_calculator)
    fermi_energy_sc = dos_calculator.get_self_consistent_fermi_energy_ev(dos_data)

    # The LDOS calculator can accept preprocessed data.
    total_energy = ldos_calculator.get_total_energy(dos_data=dos_data, density_data=dens_data, fermi_energy_eV=fermi_energy_sc)
    printout("FESL total energy: ", total_energy, " eV")
    printout("QE total energy: ", ldos_calculator.total_energy_dft_calculation, "eV")
    if np.abs(total_energy - ldos_calculator.total_energy_dft_calculation) > accuracy:
        print(np.abs(total_energy - ldos_calculator.total_energy_dft_calculation))
        return False
    printout("Parameters used for this experiment:")
    test_parameters.show()
    return True

if __name__ == "__main__":
    if run_example09():
        printout("Successfully ran ex09_get_total_energy.py.")
    else:
        raise Exception(
            "Ran ex09_get_total_energy but something was off. If you haven't changed any parameters in "
            "the example, there might be a problem with your installation.")

