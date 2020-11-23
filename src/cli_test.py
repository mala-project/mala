from common.parameters import parameters
from datahandling.data_mockup import data_mockup
from network.network import network
from network.trainer import trainer

'''
This framework should be easily usable by instantiating and calling a couple of classes.
This file is the central testing point to try out all new functions, as it serves as a command-line based testing interface.
'''

print("THIS IS ALL STILL WORK IN PROGRESS.")
print("Welcome to ML-DFT@CASUS.")

####################
# PARSING
# TODO: Add parser.
####################



####################
# PARAMETERS
# All paramaters are handled from a central parameters class that contains subclasses.
####################
test_parameters = parameters()
# Modify parameters as you want...
test_parameters.data.directory = "/home/fiedlerl/data/mnist/"
test_parameters.comment = "Test run using MNIST."
#used_parameters.network.nn_type = "A different neural network"

####################
# DATA
# Read data to numpy arrays (later we are going to use OpenPMD for this)
# For the fingerprints, snapshot creation has to take place here.
####################
data_handler = data_mockup(test_parameters)
data_handler.load_data()
data_handler.prepare_data()
data_handler.dbg_reduce_number_of_data_points(5000, 1000, 1000)
print("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
####################

test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 30, data_handler.get_output_dimension()]
test_network = network(test_parameters)
#test_trainer = trainer(test_parameters)
print("Network setup: DONE.")


print("Setup COMPLETE. \nParameters used for this experiment:")
test_parameters.show()

####################
# TRAINING
# Train the network.
####################

####################
# SAVING
# Save the network.
####################

####################
# INFERENCE
# Load a network and atomic configuration.
# Create the snapshot data and give it to the network,
# obtaining the LDOS.
####################

####################
# (optional) POSTPROCESSING
# Unclear if this can be done with this framework.
####################
