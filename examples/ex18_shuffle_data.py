import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
ex18_shuffle_data.py: Shows how data can be shuffled amongst multiple
snapshots, which is very useful in the lazy loading case, where this cannot be 
easily done in memory. 
"""

####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()
# Important for the shuffling: you can set a custom seed to replicate
# earlier experiments.
test_parameters.data.shuffling_seed = 1234

####################
# DATA
# Add data and shuffle.
####################

data_shuffler = mala.DataShuffler(test_parameters)

# Add a snapshot we want to use in to the list.
data_shuffler.add_snapshot("Be_snapshot0.in.npy", data_path,
                           "Be_snapshot0.out.npy", data_path)
data_shuffler.add_snapshot("Be_snapshot1.in.npy", data_path,
                           "Be_snapshot1.out.npy", data_path)

# After shuffling, these snapshots can be loaded as regular snapshots for
# lazily loaded training-
data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*")


