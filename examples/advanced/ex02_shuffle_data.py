import os

import mala

from mala.datahandling.data_repo import data_path

"""
Shows how data can be shuffled amongst multiple
snapshots, which is very useful in the lazy loading case, where this cannot be
easily done in memory.
"""


parameters = mala.Parameters()

# This ensures reproducibility of the created data sets.
parameters.data.shuffling_seed = 1234

data_shuffler = mala.DataShuffler(parameters)
data_shuffler.add_snapshot(
    "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path
)
data_shuffler.add_snapshot(
    "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path
)
data_shuffler.shuffle_snapshots(
    complete_save_path=".", save_name="Be_shuffled*"
)
