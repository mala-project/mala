import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")

"""
ex01_run_singleshot.py: Shows how a neural network can be trained on material
data using this framework. It uses preprocessed data, that is read in
from *.npy files.
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()
# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
test_parameters.data.data_splitting_type = "by_snapshot"
test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the used activation function.
test_parameters.network.layer_activations = ["ReLU"]

# Specify the training parameters.
test_parameters.running.max_number_epochs = 20
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                          "Al_debug_2k_nr0.out.npy", data_path,
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                          "Al_debug_2k_nr1.out.npy", data_path,
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                          "Al_debug_2k_nr2.out.npy", data_path,
                          output_units="1/Ry")
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                       100,
                                       data_handler.get_output_dimension()]

# Setup network and trainer.
test_network = mala.Network(test_parameters)
test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

printout("Starting training.")
test_trainer.train_network()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
