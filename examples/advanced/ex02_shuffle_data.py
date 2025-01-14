import mala

from mala.datahandling.data_repo import data_path

"""
Shows how data can be shuffled amongst multiple
snapshots, which is very useful in the lazy loading case, where this cannot be
easily done in memory.
"""

# There are two ways to perform shuffling. One can either shuffle snapshots
# on the fly and create snapshot-like, temporary files to be directly used
# for training, or save them to permanent files, and load them as needed.
# The latter is useful when multiple networks are to be trained with the same
# shuffled data, the former may save on hard-disk space by deleting
# shuffled data right after usage.


# 1. Shuffling snapshots on the fly.

# Define parameters for network training after shuffling.
parameters = mala.Parameters()
parameters.verbosity = 1
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "minmax"
parameters.network.layer_activations = ["ReLU"]
parameters.running.max_number_epochs = 100
parameters.running.mini_batch_size = 40
parameters.running.learning_rate = 0.00001
parameters.running.optimizer = "Adam"
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5
parameters.data.use_lazy_loading = True

# This ensures reproducibility of the created data sets.
parameters.data.shuffling_seed = 1234

data_shuffler = mala.DataShuffler(parameters)

# Instead of precomputed snapshots, on-the-fly calculated ones may be used
# as well. Here, we use precomputed ones.
data_shuffler.add_snapshot(
    "Be_snapshot0.in.npy",
    data_path,
    "Be_snapshot0.out.npy",
    data_path,
)
data_shuffler.add_snapshot(
    "Be_snapshot1.in.npy",
    data_path,
    "Be_snapshot1.out.npy",
    data_path,
)
# On-the-fly snapshots can be added as well.
# data_shuffler.add_snapshot(
#     "Be_snapshot2.info.json",
#     data_path,
#     "Be_snapshot2.out.npy",
#     data_path,
# )


# Shuffle the snapshots using the "shuffle_to_temporary" option.
data_shuffler.shuffle_snapshots(
    complete_save_path=data_path,
    save_name="Be_shuffled*",
    shuffle_to_temporary=True,
    number_of_shuffled_snapshots=2,
)

# The data shuffler provides a list with these temporary, snapshot-like
# objects that can then be used in a network training.

data_handler = mala.DataHandler(parameters)
data_handler.add_snapshot(
    data_shuffler.temporary_shuffled_snapshots[0].input_npy_file,
    data_shuffler.temporary_shuffled_snapshots[0].input_npy_directory,
    data_shuffler.temporary_shuffled_snapshots[0].output_npy_file,
    data_shuffler.temporary_shuffled_snapshots[0].output_npy_directory,
    "tr",
)
data_handler.add_snapshot(
    data_shuffler.temporary_shuffled_snapshots[1].input_npy_file,
    data_shuffler.temporary_shuffled_snapshots[1].input_npy_directory,
    data_shuffler.temporary_shuffled_snapshots[1].output_npy_file,
    data_shuffler.temporary_shuffled_snapshots[1].output_npy_directory,
    "va",
)
data_handler.prepare_data()

# Nowe we can train a network in the standard MALA way.
parameters.network.layer_sizes = [
    data_handler.input_dimension,
    100,
    data_handler.output_dimension,
]
network = mala.Network(parameters)
trainer = mala.Trainer(parameters, network, data_handler)
trainer.train_network()

# Afterwards, the temporary files should be deleted.
data_shuffler.delete_temporary_shuffled_snapshots()

# 2. Shuffling snapshots to permanent files.
# After shuffling, the standard approach to MALA data loading can/should
# be used to train a network.

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
