from fesl.common.parameters import Parameters
from fesl.descriptors.snap import SNAP
import numpy as np

"""
ex05_create_snap_descriptors.py: Shows how the framework can be used to create SNAP descriptors, from a
Quantum Espresso runfile. No training will/has to be performed for this experiment. 
"""

print("Welcome to FESL.")
print("Running ex05_create_snap_descriptors.py")

####################
# PARAMETERS
# We only have to specify the parameters relevant to the descriptors.
# We do, however, define a smaller debug grid as to make sure that this calculation does not take
# too long.
####################
test_parameters = Parameters()
test_parameters.descriptors.descriptor_type = "SNAP"
test_parameters.descriptors.twojmax = 10
test_parameters.descriptors.rcutfac = 4.67637
test_parameters.debug.grid_dimensions = [20, 10, 10]

####################
# DESCRIPTORS
# We use a SNAP object to calculate the SNAP descriptors from a QE file.
####################

test_descriptors = SNAP(test_parameters)
snap_data = test_descriptors.calculate_from_qe_out("QE_Al.scf.pw.out", "./data/")

# Optional: Verify that the we got the correct results.
do_check = True
if do_check:
    snap_data_backup = np.load("./data/Al_snap_from_qe.npy")
    diff = np.abs(np.sum(snap_data_backup) - np.sum(snap_data))
    print(diff)
    if diff > 0.00001:
        print("SNAP descriptors are NOT matching.")
    else:
        print("SNAP descriptors ARE matching.")

print("Successfully ran ex05_create_snap_descriptors.py.")
