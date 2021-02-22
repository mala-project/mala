from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.datahandling.data_handler import DataHandler
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()

"""
ex03_run_singleshot.py: Demonstrate the file saving capabilites of a data loader to resize a snapshot for debugging
purposes.
"""
def run_example03(accuracy=10**-10):
    printout("Welcome to FESL.")
    printout("Running ex03_run_singleshot.py")

    ####################
    # PARAMETERS
    # We only need to specify the correct input data format and the resizing grid.
    ####################
    test_parameters = Parameters()

    # We want to resize certain data files, so we don't want any kind of cutting operation performed.
    test_parameters.data.descriptors_contain_xyz = False
    test_parameters.debug.grid_dimensions = [200, 10, 1]

    ####################
    # DATA INPUT
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    ####################

    data_handler = DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0.npy",
                              data_path+"Al256/",
                              "Al_ldos_200x200x200grid_250elvls_snapshot0.npy",
                              data_path+"Al256/",output_units="1/Ry")
    data_handler.resize_snapshots_for_debugging(directory="./data/",
                                                naming_scheme_input="test_Al_debug_2k_nr*.in",
                                                naming_scheme_output="test_Al_debug_2k_nr*.out")
    printout("Write data: DONE.")

    # Verify that the we got the correct results.
    # Since we work with float32 internally, there will be some difference, that will scale with
    # data size. So we need to divide by the amount of data.
    data_backup = np.load(data_path+"Al256_reduced/"+"Al_debug_2k_nr0.in.npy")
    data_calculated = np.load("./data/test_Al_debug_2k_nr0.in.npy")
    diffin = (np.abs(np.sum(data_backup) - np.sum(data_calculated)))/\
             (np.sum(data_backup) *test_parameters.debug.grid_dimensions[0]*
              test_parameters.debug.grid_dimensions[1]*test_parameters.debug.grid_dimensions[2])

    data_backup = np.load(data_path+"Al256_reduced/"+"Al_debug_2k_nr0.out.npy")
    data_calculated = np.load("./data/test_Al_debug_2k_nr0.out.npy")
    diffout = (np.abs(np.sum(data_backup) - np.sum(data_calculated)))/\
             (np.sum(data_backup)*test_parameters.debug.grid_dimensions[0]*
              test_parameters.debug.grid_dimensions[1]*test_parameters.debug.grid_dimensions[2])

    if diffin > accuracy or diffout > accuracy:
        return False
    else:
        return True


if __name__ == "__main__":
    if run_example03():
        printout("Successfully ran ex03_run_singleshot.py.")
    else:
        raise Exception("Ran ex03_run_singleshot but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")
