import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex01_train_network.py: Shows how a neural network can be trained on material
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
# done on a "by snapshot" basis.
test_parameters.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the used activation function.
test_parameters.network.layer_activations = ["ReLU"]

# Specify the training parameters.
test_parameters.running.max_number_epochs = 100
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"
test_parameters.verbosity = 1

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                          "Be_snapshot0.out.npy", data_path, "tr")
data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                          "Be_snapshot1.out.npy", data_path, "va")
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

test_parameters.network.layer_sizes = [data_handler.input_dimension,
                                       100,
                                       data_handler.output_dimension]

# Setup network and trainer.
test_network = mala.Network(test_parameters)
test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network and save it afterwards.
####################

printout("Starting training.")
test_trainer.train_network()

# Additional calculation data, i.e., a simulation output, may be saved
# along side the model. This makes future inference easier.
additional_calculation_data = os.path.join(data_path, "Be_snapshot0.out")
test_trainer.save_run("be_model",
                      additional_calculation_data=additional_calculation_data)
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
