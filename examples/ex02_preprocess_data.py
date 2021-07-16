import mala
from mala import printout
from data_repo_path import get_data_repo_path
import numpy as np
data_path = get_data_repo_path()+"Al36/"

"""
ex02_preprocess_data.py: Shows how this framework can be used to preprocess
data. Preprocessing here means converting raw DFT calculation output into 
numpy arrays of the correct size. For the input data, this means descriptor
calculation.

Further preprocessing steps (scaling, unit conversion) is done later. 
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()

# Specify input data options, i.e. which descriptors are calculated
# with which parameters. These are the standard parameters for
# the calculation of SNAP descriptors.
test_parameters.descriptors.descriptor_type = "SNAP"
test_parameters.descriptors.twojmax = 10
test_parameters.descriptors.rcutfac = 4.67637
test_parameters.data.descriptors_contain_xyz = True

# Specify output data options, i.e. how the LDOS is parsed.
# The Al system used as an example here actually has 250 energy levels.
# But for the convenience of the user, only 10 energy levels will be
# used for this example.
test_parameters.targets.target_type = "LDOS"
test_parameters.targets.ldos_gridsize = 10
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10

####################
# DATA
# Create a DataConverter, and add snapshots to it.
####################

data_converter = mala.DataConverter(test_parameters)

# Take care to choose the "add_snapshot" function correct for
# the type of data you want to preprocess.
data_converter.add_snapshot_qeout_cube("Al.pw.scf.out", data_path,
                                       "cubes/tmp.pp*Al_ldos.cube",
                                       data_path, output_units="1/Ry")

# Convert all the snapshots and save them in the current directory.
data_converter.convert_snapshots("./", naming_scheme="Al_snapshot*")

####################
# RESULTS.
# Print the used parameters and check whether the preprocessed data
# has the desired dimensions.
####################

printout("Parameters used for this experiment:")
test_parameters.show()