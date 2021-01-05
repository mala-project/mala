from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.network.network import Network
from fesl.network.trainer import Trainer

"""
ex01_run_singleshot.py: Shows how a neural network can be trained on material data using this framework.
It uses preprocessed data, that is read in from *.npy files.
"""

print("Welcome to FESL.")
print("Running ex01_run_singleshot.py")

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
test_parameters.data.datatype_in = "*.npy"
test_parameters.data.datatype_out = "*.npy"
test_parameters.data.data_splitting_type = "random"
test_parameters.data.data_splitting_percent = [80, 10, 10]
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"
test_parameters.descriptors.twojmax = 11
test_parameters.targets.ldos_gridsize = 10
test_parameters.network.layer_activations = ["ReLU"]
test_parameters.training.max_number_epochs = 20
test_parameters.training.mini_batch_size = 40
test_parameters.training.learning_rate = 0.00001
test_parameters.training.trainingtype = "Adam"

####################
# DATA
# Read data into RAM.
# We have to specify the directories we want to read the snapshots from.
# The Handlerinterface will also return input and output scaler objects. These are used internally to scale
# the data. The objects can be used after successful training for inference or plotting.
####################

data_handler = HandlerInterface(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/", output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", "./data/", "Al_debug_2k_nr1.out.npy", "./data/", output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")

data_handler.load_data()
data_handler.prepare_data()
print("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
####################


test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

# Setup network and trainer.
test_network = Network(test_parameters)
test_trainer = Trainer(test_parameters)
print("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

print("Starting training.")
test_trainer.train_network(test_network, data_handler)
print("Training: DONE.")

####################
print("Successfully ran ex01_run_singleshot.py.")
print("Parameters used for this experiment:")
test_parameters.show()
