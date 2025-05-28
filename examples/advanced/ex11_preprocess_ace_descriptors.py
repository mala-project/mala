import os

import mala

from mala.datahandling.data_repo import data_path_be

"""
Shows how ACE descriptors can be calculated with MALA. ACE descriptors 
hold the promise of being more descriptive and accurate than bispectrum 
descriptors and are currently being investigated by the MALA team. 
MALA already implements most functionalities of bispectrum descriptors for 
ACE descriptors. 
After preprocessing data to ACE descriptors with this example, all other 
examples can be run with the ACE descriptors by simply pointing to the 
appropriate ACE descriptor numpy files.

REQUIRES LAMMPS.
"""


####################
# 1. PARAMETERS
# Compare to the bispectrum descriptors, ACE descriptors have slightly more
# hyperparameters to consider. For more information on the hyperparameters,
# please refer to https://arxiv.org/abs/2411.19617.
# ACE_DOCS_MISSING: Is it enough to refer to the paper here or should
# we include more information here?
####################

parameters = mala.Parameters()
# Bispectrum parameters.
parameters.descriptors.descriptor_type = "ACE"
parameters.descriptors.ace_included_expansion_ranks = [1, 2, 3]
parameters.descriptors.ace_maximum_l_per_rank = [0, 1, 1]
parameters.descriptors.ace_maximum_n_per_rank = [1, 1, 1]
parameters.descriptors.ace_minimum_l_per_rank = [0, 0, 0]

####################
# 2. ADDING DATA FOR DATA CONVERSION
# As detailed in the basic preprocessing example, we can process
# input and output data separately. This example only computes descriptor data
# as it is assumed that the LDOS data has already been processed in an earlier
# example.
####################

data_converter = mala.DataConverter(parameters)
outfile = os.path.join(data_path_be, "Be_snapshot0.out")

# Converting a snapshot for training on precomputed descriptor data.
data_converter.add_snapshot(
    descriptor_input_type="espresso-out",
    descriptor_input_path=outfile,
)

####################
# 3. Converting the data
# Since we are only converting descriptor data here, target and simulation
# output paths can be ommitted.
####################

data_converter.convert_snapshots(
    descriptor_save_path="./",
    naming_scheme="Be_snapshot*_ACE.npy",
    descriptor_calculation_kwargs={"working_directory": data_path_be},
)
