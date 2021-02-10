from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.descriptors.snap import SNAP
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()

"""
ex05_create_snap_descriptors.py: Shows how the framework can be used to create SNAP descriptors, from a
Quantum Espresso runfile. No training will/has to be performed for this experiment. 
"""
def run_example05(accuracy = 0.000001):
    printout("Welcome to FESL.")
    printout("Running ex05_create_snap_descriptors.py")

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
    snap_data = test_descriptors.calculate_from_qe_out("QE_Al.scf.pw.out", data_path+"Al256/")
    snap_data_backup = np.load(data_path+"Al256_reduced/"+"Al_snap_from_qe.npy")
    diff = np.abs(np.sum(snap_data_backup) - np.sum(snap_data))
    if diff > accuracy:
        return False
    test_parameters.show()
    return True

if __name__ == "__main__":
    if run_example05():
        printout("Successfully ran ex05_create_snap_descriptors.")
    else:
        raise Exception(
            "Ran ex05_create_snap_descriptors but something was off. If you haven't changed any parameters in "
            "the example, there might be a problem with your installation.")
