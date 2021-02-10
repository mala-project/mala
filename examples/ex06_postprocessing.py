from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.targets.target_interface import TargetInterface
import matplotlib.pyplot as plt
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"

"""
ex06_postprocessing.py: Shows how the LDOS can be post processed. 
"""

def run_example06(doplots = True, accuracy = 0.00001):
    printout("Welcome to FESL.")
    printout("Running ex06_postprocessing.py")
    printout("The physical quantities calculated in this example will seem wrong, since only a fraction of a DFT simulation "
          "cell is used.")

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = Parameters()
    test_parameters.targets.target_type = "LDOS"
    test_parameters.targets.ldos_gridsize = 250
    test_parameters.targets.ldos_gridspacing_ev = 0.1
    test_parameters.targets.ldos_gridoffset_ev = -10

    ####################
    # DATA
    # Read data into RAM.
    # For this test we can simply read in the numpy arrays and convert them using the LDOS class.
    ####################

    # Our target data is LDOS data.
    ldos = TargetInterface(test_parameters)

    # We load additional information about the data using the output of the actual quantum espresso calculation.
    ldos.read_additional_calculation_data("qe.out", data_path+"QE_Al.scf.pw.out")

    # The LDOS data needs to be converted; we got it from a QE calculation.
    ldos_data = ldos.convert_units(np.load(data_path+"Al_debug_2k_nr0.out.npy"), in_units="1/Ry")

    #####
    # Density
    #####

    dens = ldos.get_density(ldos_data, conserve_dimensions=True, integration_method="analytical")
    # For plotting we use the fact that the reduced datasets have dimensions of 200x10x1.
    if doplots:
        x_range = np.linspace(0, ldos.grid_spacing_Bohr*np.shape(dens)[0], np.shape(dens)[0])
        for i in range(0, np.shape(dens)[1]):
            y_iso_line = []
            for j in range(0, np.shape(dens)[0]):
                y_iso_line.append(dens[j, i, 0])
            descr = "y = "+str(i)
            plt.plot(x_range, y_iso_line, label=descr)
        plt.xlabel("x [Bohr]")
        plt.ylabel("n [a.u.]")
        plt.legend()
        plt.show()

    #####
    # Density of states
    #####

    dos = ldos.get_density_of_states(ldos_data, integration_method="summation")
    if doplots:
        e_range = np.linspace(test_parameters.targets.ldos_gridoffset_ev, test_parameters.targets.ldos_gridoffset_ev+
                              test_parameters.targets.ldos_gridsize*test_parameters.targets.ldos_gridspacing_ev,
                              test_parameters.targets.ldos_gridsize)
        plt.plot(e_range, dos, label="DOS")
        plt.xlabel("E [eV]")
        plt.ylabel("D(E) [1/eV]")
        plt.legend()
        plt.show()

    #####
    # Band energies
    #####

    e_band = ldos.get_band_energy(ldos_data)
    printout("Band energy [eV]: ", e_band)
    # Result should be around 0.5516523883883581
    if np.abs(0.5516523883883581 - e_band) > accuracy:
        return False

    #####
    # Number of electrons
    #####

    nr_electrons = ldos.get_number_of_electrons(ldos_data)
    printout("# of Electrons: ", nr_electrons)
    # Result should be around 0.16438792929832854
    if np.abs(0.16438792929832854 - nr_electrons) > accuracy:
        return False

    printout("Successfully ran ex06_postprocessing.py.")
    printout("Parameters used for this experiment:")
    test_parameters.show()
    return True

if __name__ == "__main__":
    if run_example06():
        printout("Successfully ran ex06_postprocessing.py.")
    else:
        raise Exception("Ran ex06_postprocessing but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")

