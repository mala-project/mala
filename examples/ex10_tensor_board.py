import os
import shutil
import webbrowser

import mala
from mala import printout
from tensorboard import program

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")


"""
ex10_tensor_board.py: Shows how a NN training by MALA can be visualized using 
tensorboard. The training is the similar to the one performed in ex01, but 
at the end of it, tensorboard will be launched in your browser.
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
test_parameters.running.max_number_epochs = 400
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.001
test_parameters.running.trainingtype = "Adam"

# Turn the visualization on and select a folder to save the visualization
# files into.
test_parameters.running.visualisation = 1
test_parameters.running.visualisation_dir = "ex10_vis"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                          "Al_debug_2k_nr0.out.npy", data_path, "tr",
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                          "Al_debug_2k_nr1.out.npy", data_path, "va",
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                          "Al_debug_2k_nr2.out.npy", data_path, "te",
                          output_units="1/Ry")
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

printout("Run finished, launch tensorboard with \"tensorboard --logdir " +
         test_trainer.full_visualization_path+"\"")
