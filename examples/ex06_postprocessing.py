from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.targets.target_interface import TargetInterface
import matplotlib.pyplot as plt
import numpy as np
"""
ex06_postprocessing.py: Shows how the LDOS can be post processed. 
"""

print("Welcome to FESL.")
print("Running ex06_postprocessing.py")
print("The physical quantities calculated in this example will seem wrong, since only a fraction of a DFT simulation "
      "cell is used.")

# Set to true to see plots.
doplots = True

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
test_parameters.targets.ldos_gridsize = 250
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10


####################
# DATA
# Read data into RAM.
# We have to specify the directories we want to read the snapshots from.
# The Handlerinterface will also return input and output scaler objects. These are used internally to scale
# the data. The objects can be used after successful training for inference or plotting.
####################

data_handler = HandlerInterface(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/")

# Our target data is LDOS data.
ldos = TargetInterface(test_parameters)

# We load additional information about the data using the output of the actual quantum espresso calculation.
ldos.read_additional_calculation_data("qe.out", "./data/QE_Al.scf.pw.out")

# Data is loaded.
data_handler.load_data()

#####
# Density
#####

dens = ldos.get_density(data_handler.raw_output_grid[0], conserve_dimensions=True, integration_method="analytical")
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

dos = ldos.get_density_of_states(data_handler.raw_output_grid[0])
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

e_band = ldos.get_band_energy(data_handler.raw_output_grid[0])
print("Band energy [eV]: ", e_band)


#####
# Number of electrons
#####

nr_electrons = ldos.get_number_of_electrons(data_handler.raw_output_grid[0])
print("# of Electrons: ", nr_electrons)

print("Successfully ran ex06_postprocessing.py.")
print("Parameters used for this experiment:")
test_parameters.show()
