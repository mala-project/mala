from common.parameters import parameters
from datahandling.handler_interface import handler_interface
from descriptors.descriptor_interface import descriptor_interface
from targets.target_interface import target_interface
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
test_parameters.data.datatype_in = "qe.out"
test_parameters.data.datatype_out = "*.cube"
test_parameters.targets.ldos_gridsize = 10
test_parameters.descriptors.dbg_grid_dimensions = [20,20,20]
test_parameters.descriptors.twojmax = 11
test_parameters.training.max_number_epochs = 10
test_parameters.training.mini_batch_size = 10
test_parameters.training.learning_rate = 3
test_parameters.comment = "Test run of ML-DFT@CASUS."
#used_parameters.network.nn_type = "A different neural network"

####################
# DATA
# Read data into RAM.
# To read and store data we need to create a descriptor calculator and a target parser.
# The former makes sure that descriptors (input quantity) are correctly read/calculated.
# The latter makes sure that targets (output quantity) are correctly read (maybe calculated, but mostly read).
# We also have to specify the directories we want to read the snapshots from.
####################
descriptor_calculator = descriptor_interface(test_parameters)
target_parser = target_interface(test_parameters)
data_handler = handler_interface(test_parameters, descriptor_calculator=descriptor_calculator, target_parser=target_parser)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("QE_Al.scf.pw.out", "/home/fiedlerl/data/test_fp_snap/0.4gcc/", "tmp.pp*Al_ldos.cube", "/home/fiedlerl/data/test_fp_snap/0.4gcc/")
data_handler.load_data()
data_handler.prepare_data()

# data_handler.prepare_data(less_training_pts=5000, less_validation_points=1000, less_test_points=1000)
# data_handler.prepare_data(less_training_pts=0, less_validation_points=0, less_test_points=0)
print("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
####################

test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 30, 30, 30, data_handler.get_output_dimension()]
test_parameters.network.layer_activations = ["Sigmoid"]
test_network = network(test_parameters)
test_trainer = trainer(test_parameters)
print("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

print("Starting training.")
test_trainer.train_network(test_network, data_handler)

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


####################
# Output parameters (=metadata)
####################

print("Execution complete. \nParameters used for this experiment:")
test_parameters.show()
