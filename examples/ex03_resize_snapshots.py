from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface

"""
ex03_run_singleshot.py: Demonstrate the file saving capabilites of a data loader to resize a snapshot for debugging
purposes.
"""

print("Welcome to FESL.")
print("Running ex03_run_singleshot.py")

####################
# PARAMETERS
# We only need to specify the correct input data format and the resizing grid.
####################
test_parameters = Parameters()
test_parameters.data.datatype_in = "*.npy"
test_parameters.data.datatype_out = "*.npy"
test_parameters.data.input_memmap_mode = "r"
test_parameters.data.output_memmap_mode = "r"

# We want to resize certain data files, so we don't want any kind of cutting operation performed.
test_parameters.data.descriptors_contain_xyz = False
test_parameters.debug.grid_dimensions = [200, 10, 1]

####################
# DATA INPUT
# Read data into RAM.
# We have to specify the directories we want to read the snapshots from.
####################

data_handler = HandlerInterface(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",
                          "Al_ldos_200x200x200grid_250elvls_snapshot0.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",output_units="1/Ry")
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot1.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",
                          "Al_ldos_200x200x200grid_250elvls_snapshot1.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",output_units="1/Ry")
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot2.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",
                          "Al_ldos_200x200x200grid_250elvls_snapshot2.npy",
                          "/home/fiedlerl/data/test_fp_snap/2.699gcc/",output_units="1/Ry")
data_handler.load_data()
print("Read data: DONE.")

####################
# DATA OUTPUT
# We save the "unprepared" data into *.npy files.
####################

data_handler.save_loaded_data(directory="/home/fiedlerl/data/test_fp_snap/2.699gcc/", filetype="*.npy",
                              naming_scheme_input="Al_debug_2k_nr*.in", naming_scheme_output="Al_debug_2k_nr*.out")
print("Write data: DONE.")
print("Successfully ran ex03_run_singleshot.py.")
