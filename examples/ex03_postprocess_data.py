import fesl
from fesl import printout
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"


"""
ex03_postprocess_data.py: Shows how this framework can be used to
postprocess data. Usually, this framework outputs LDOS data, thefore,
post processing of LDOS data will be shown in the following. 
Set do_total_energy to False, if you don't have the QuantumEspresso
Python module installed.
"""


def run_example03(do_total_energy=True, accuracy_electrons=1e-11,
                  accuracy_total_energy=50):

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################
    test_parameters = fesl.Parameters()

    # Specify the correct LDOS parameters.
    test_parameters.targets.target_type = "LDOS"
    test_parameters.targets.ldos_gridsize = 250
    test_parameters.targets.ldos_gridspacing_ev = 0.1
    test_parameters.targets.ldos_gridoffset_ev = -10

    ####################
    # TARGETS
    # Create a target calculator to postprocess data.
    # Use this calculator to perform various operations.
    ####################

    ldos = fesl.TargetInterface(test_parameters)

    # Read additional information about the calculation.
    # By doing this, the calculator is able to know e.g. the temperature
    # at which the calculation took place or the lattice constant used.
    ldos.read_additional_calculation_data("qe.out",
                                          data_path+"Al.pw.scf.out")

    # Read in LDOS data. For actual workflows, this part will come
    # from a network.
    ldos_data = np.load(data_path+"Al_ldos.npy")

    # Get quantities of interest.
    # For better values in the post processing, it is recommended to
    # calculate the "self-consistent Fermi energy", i.e. the Fermi energy
    # at which the (L)DOS reproduces the exact number of electrons.
    # This Fermi energy usually differs from the one outputted by the
    # QuantumEspresso calculation, due to numerical reasons. The difference
    # is usually very small.
    self_consistent_fermi_energy = ldos.\
        get_self_consistent_fermi_energy_ev(ldos_data)
    number_of_electrons = ldos.\
        get_number_of_electrons(ldos_data, fermi_energy_eV=
                                self_consistent_fermi_energy)
    band_energy = ldos.get_band_energy(ldos_data,
                                       fermi_energy_eV=
                                       self_consistent_fermi_energy)
    total_energy = 0.0
    if do_total_energy:
        # To perform a total energy calculation one also needs to provide
        # a pseudopotential(path).
        ldos.set_pseudopotential_path(data_path)
        total_energy = ldos.get_total_energy(ldos_data,
                                             fermi_energy_eV=
                                             self_consistent_fermi_energy)

    ####################
    # RESULTS.
    # Print the used parameters and check whether LDOS based results
    # are consistent with the actual DFT results.
    ####################

    printout("Parameters used for this experiment:")
    test_parameters.show()

    print("Number of electrons:", number_of_electrons)
    print("Band energy:", band_energy)
    if do_total_energy:
        print("Total energy:", total_energy)

    if np.abs(number_of_electrons - ldos.number_of_electrons) > \
            accuracy_electrons:
        return False

    # FIXME: Add  as soon as band_energy_dft_calculation is fixed.
    # if np.abs(number_of_electrons - ldos.number_of_electrons) > accuracy:
    #     return True

    if do_total_energy:
        if np.abs(total_energy - ldos.total_energy_dft_calculation) > \
                accuracy_total_energy:
            return False
    return True


if __name__ == "__main__":
    if run_example03():
        printout("Successfully ran ex03_postprocess_data.")
    else:
        raise Exception("Ran ex03_postprocess_data but something was off."
                        " If you haven't changed any parameters in "
                        "the example, there might be a problem with your"
                        " installation.")
