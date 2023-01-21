import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex09_distributed_hyperopt.py: Shows how a hyperparameter 
optimization can be sped up using a RDB storage. Ideally this should be done
using a database server system, such as PostgreSQL or MySQL. 
For this easy example, sqlite will be used. It is highly advisory not to
to use this for actual, at-scale calculations!

REQUIRES SQLITE.
"""


# Remove the previous data base.
try:
    os.remove("ex09.db")
except FileNotFoundError:
    pass

####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################
test_parameters = mala.Parameters()
# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis.
test_parameters.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the training parameters.
test_parameters.running.max_number_epochs = 5
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"

# Specify the number of trials, the hyperparameter optimizer should run
# and the type of hyperparameter.
test_parameters.hyperparameters.n_trials = 10
test_parameters.hyperparameters.checkpoints_each_trial = -1
test_parameters.hyperparameters.checkpoint_name = "ex09"
test_parameters.hyperparameters.hyper_opt_method = "optuna"
test_parameters.hyperparameters.study_name = "ex09"
test_parameters.hyperparameters.rdb_storage = 'sqlite:///ex09.db'

# Hyperparameter optimization can be further refined by using ensemble training
# at each step and by using a different metric then the validation loss
# (e.g. the band energy). It is recommended not to use the ensemble training
# method in Single-GPU use, as it naturally drastically decreases performance.
test_parameters.targets.ldos_gridsize = 11
test_parameters.targets.ldos_gridspacing_ev = 2.5
test_parameters.targets.ldos_gridoffset_ev = -5
test_parameters.hyperparameters.number_training_per_trial = 3
test_parameters.running.after_before_training_metric = "band_energy"

####################
# DATA
# Add and prepare snapshots for training.
####################
data_handler = mala.DataHandler(test_parameters)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                          "Be_snapshot1.out.npy", data_path, "tr",
                          calculation_output_file=
                          os.path.join(data_path, "Be_snapshot1.out"))
data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                          "Be_snapshot2.out.npy", data_path, "va",
                          calculation_output_file=
                          os.path.join(data_path, "Be_snapshot2.out"))
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# HYPERPARAMETER OPTIMIZATION
# In order to perform a hyperparameter optimization,
# one has to simply create a hyperparameter optimizer
# and let it perform a "study".
# Before such a study can be done, one has to add all the parameters
# of interest.
####################

test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)

# Learning rate will be optimized.
test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                     0.0000001, 0.01)

# Number of neurons per layer will be optimized.
test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)

# Choices for activation function at each layer will be optimized.
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00",
                                     choices=["ReLU", "Sigmoid"])
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01",
                                     choices=["ReLU", "Sigmoid"])
test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02",
                                     choices=["ReLU", "Sigmoid"])

# Perform hyperparameter optimization.
printout("Starting Hyperparameter optimization.")
test_hp_optimizer.perform_study()
test_hp_optimizer.set_optimal_parameters()
printout("Hyperparameter optimization: DONE.")

####################
# TRAINING
# Train with these new parameters.
####################

test_network = mala.Network(test_parameters)
test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
printout("Network setup: DONE.")
test_trainer.train_network()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
