from common.parameters import Parameters
from datahandling.handler_interface import HandlerInterface
from descriptors.descriptor_interface import DescriptorInterface
from targets.target_interface import TargetInterface
from network.network import Network
from network.trainer import Trainer
from datahandling.data_scaler import DataScaler
'''
This framework should be easily usable by instantiating and calling a couple of classes.
This file is the central testing point to try out all new functions, as it serves as a command-line based testing interface.
'''

print("THIS IS ALL STILL WORK IN PROGRESS.")
print("Welcome to ML-DFT@CASUS.")

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
# Modify parameters as you want...
test_parameters.data.datatype_in = "*.npy"
test_parameters.data.datatype_out = "*.npy"
test_parameters.data.data_splitting_percent = [80, 10, 10]

test_parameters.scaling.input_rescaling_type = "feature-wise-standard"
test_parameters.scaling.output_rescaling_type = "normal"

test_parameters.descriptors.twojmax = 11

test_parameters.targets.ldos_gridsize = 10

test_parameters.training.max_number_epochs = 20
test_parameters.training.mini_batch_size = 10
test_parameters.training.learning_rate = 7

test_parameters.comment = "Test run of ML-DFT@CASUS."
# test_parameters.debug.grid_dimensions = [200,100,1]

####################
# DATA
# Read data into RAM.
# To read and store data we need to create a descriptor calculator and a target parser.
# The former makes sure that descriptors (input quantity) are correctly read/calculated.
# The latter makes sure that targets (output quantity) are correctly read (maybe calculated, but mostly read).
# We also need a data_scaler, that will make sure the data is correctly scaled (both for training and inference).
# We also have to specify the directories we want to read the snapshots from.
####################
descriptor_calculator = DescriptorInterface(test_parameters)
target_parser = TargetInterface(test_parameters)
data_scaler = DataScaler(test_parameters)
data_handler = HandlerInterface(test_parameters, descriptor_calculator=descriptor_calculator,
                                target_parser=target_parser, data_scaler=data_scaler)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0small.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/",
"Al_ldos_200x200x200grid_250elvls_snapshot0small.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/")
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0small_copy.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/",
"Al_ldos_200x200x200grid_250elvls_snapshot0small_copy.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/")
data_handler.load_data()
data_handler.prepare_data()
print("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
####################

test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]
test_parameters.network.layer_activations = ["Sigmoid"]
test_network = Network(test_parameters)
test_trainer = Trainer(test_parameters)
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
