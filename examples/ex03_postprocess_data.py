import os

import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")


"""
ex03_postprocess_data.py: Shows how this framework can be used to
postprocess data. Usually, this framework outputs LDOS data, thefore,
post processing of LDOS data will be shown in the following. 
Set do_total_energy to False, if you don't have the QuantumEspresso
Python module installed.
Since actual LDOS files are rather larger, this uses a drastically reduced
LDOS. Therefore the accuracy of the overall result is rather poor. 

REQUIRES QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""

# If you don't have the total energy module installed,
# just set the following variable to False.
do_total_energy = True

####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################
test_parameters = mala.Parameters()

# Specify the correct LDOS parameters.
test_parameters.targets.target_type = "LDOS"
test_parameters.targets.ldos_gridsize = 11
test_parameters.targets.ldos_gridspacing_ev = 2.5
test_parameters.targets.ldos_gridoffset_ev = -5
# To perform a total energy calculation one also needs to provide
# a pseudopotential(path).
test_parameters.targets.pseudopotential_path = data_path

####################
# TARGETS
# Create a target calculator to postprocess data.
# Use this calculator to perform various operations.
####################
# Read in LDOS data. For actual workflows, this part will come
# from a network.
ldos = mala.LDOS.from_numpy_file(test_parameters,
                                 os.path.join(data_path, "Be_ldos.npy"))

# Read additional information about the calculation.
# By doing this, the calculator is able to know e.g. the temperature
# at which the calculation took place or the lattice constant used.
ldos.read_additional_calculation_data("qe.out", os.path.join(
                                      data_path, "Be.pw.scf.out"))

# Get quantities of interest.
# For better values in the post processing, it is recommended to
# calculate the "self-consistent Fermi energy", i.e. the Fermi energy
# at which the (L)DOS reproduces the exact number of electrons.
# This Fermi energy usually differs from the one outputted by the
# QuantumEspresso calculation, due to numerical reasons. The difference
# is usually very small.
self_consistent_fermi_energy = ldos.fermi_energy
band_energy = ldos.band_energy
number_of_electrons = ldos.number_of_electrons
total_energy = 0.0
if do_total_energy:
    total_energy = ldos.total_energy

####################
# RESULTS.
# Print the used parameters and check whether LDOS based results
# are consistent with the actual DFT results.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
print("Values from LDOS: ")
print("Number of electrons:", number_of_electrons)
print("Band energy:", band_energy)
if do_total_energy:
    print("Total energy:", total_energy)

print("Values from DFT: ")
print("Number of electrons:", ldos.number_of_electrons_exact)
print("Band energy:", ldos.band_energy_dft_calculation)
if do_total_energy:
    print("Total energy:", ldos.total_energy_dft_calculation)

