import os

import mala
import numpy as np
from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

"""
ex16_calculate_acsd.py: Shows how MALA can be used to calculate the ACSD,
the average cosine similarity distance, which can be useful in correctly 
preprocessing data.
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()

# Specify if input data is expected to contain raw coordinates, and the
# number of points used by ACSD.
test_parameters.descriptors.descriptors_contain_xyz = True
test_parameters.descriptors.acsd_points = 100


####################
# DESCRIPTORS
# Create a Descriptors object, load the data, and calculate the ACSD.
####################
descriptors = mala.DescriptorInterface(test_parameters)
snap_data = np.load(os.path.join(data_path, "Be_snapshot1.in.npy"))
ldos_data = np.load(os.path.join(data_path, "Be_snapshot1.out.npy"))
print(descriptors.get_acsd(snap_data, ldos_data))
test_parameters.show()
