from mldft.common.parameters import Parameters
from mldft.datahandling.handler_interface import HandlerInterface
from mldft.descriptors.descriptor_interface import DescriptorInterface
from mldft.targets.target_interface import TargetInterface
from mldft.network.network import Network
from mldft.network.trainer import Trainer
from mldft.network.hyperparameter_optimizer import HyperparameterOptimizer
from mldft.network.optuna_parameter import OptunaParameter

"""
ex3_development.py: development run script.
"""

print("Welcome to ML-DFT@CASUS.")
print("Running ex3_development_script.py")

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
# Modify parameters as you want...
test_parameters.data.datatype_in = "*.npy"
test_parameters.data.datatype_out = "*.npy"
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

test_parameters.hyperparameters.n_trials = 20

test_parameters.comment = "Test run of ML-DFT@CASUS."
# test_parameters.debug.grid_dimensions = [200,100,1]

####################
# DATA
# Read data into RAM.
# To read and store data we need to create a descriptor calculator and a target parser.
# The former makes sure that descriptors (input quantity) are correctly read/calculated.
# The latter makes sure that targets (output quantity) are correctly read (maybe calculated, but mostly read).
# We also have to specify the directories we want to read the snapshots from.
# The Handlerinterface will also return input and output scaler objects. These are used internally to scale
# the data. The objects can be used after successful training for inference or plotting.
####################
descriptor_calculator = DescriptorInterface(test_parameters)
target_parser = TargetInterface(test_parameters)
data_handler, input_scaler, output_scaler = HandlerInterface(test_parameters, descriptor_calculator=descriptor_calculator, target_parser=target_parser)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0small.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/",
"Al_ldos_200x200x200grid_250elvls_snapshot0small.npy",
"/home/fiedlerl/data/test_fp_snap/2.699gcc/")
# data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot0small_copy.npy",
# "/home/fiedlerl/data/test_fp_snap/2.699gcc/",
# "Al_ldos_200x200x200grid_250elvls_snapshot0small_copy.npy",
# "/home/fiedlerl/data/test_fp_snap/2.699gcc/")
data_handler.load_data()
data_handler.prepare_data()
print("Read data: DONE.")

####################
# (optional) HYPERPARAMETER OPTIMIZATION
# In order to perform a hyperparameter optimization,
# one has to simply create a hyperparameter optimizer
# and let it perform a "study".
# This class is nothing more than a wrapper to optuna.
# Via the hyperparameter_list one can add hyperparameters to be optimized.
####################

# Add hyperparameters we want to have optimized to the list.
test_parameters.hyperparameters.hlist.append(OptunaParameter("float", "learning_rate", 0.0000001, 0.01))
test_parameters.hyperparameters.hlist.append(OptunaParameter("int", "number_of_hidden_neurons", 10, 100))
test_parameters.hyperparameters.hlist.append(OptunaParameter("categorical", "layer_activations", choices=["ReLU", "Sigmoid"]))

# Perform hyperparameter optimization.
print("Starting Hyperparameter optimization.")
test_hp_optimizer = HyperparameterOptimizer(test_parameters)
test_hp_optimizer.perform_study(data_handler)

# We can now set the optimal parameters determined in the hyperparameter optimization and train again.
print("Hyperparameter optimization: DONE.")
test_hp_optimizer.set_optimal_parameters()

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
####################

# Uncomment the next line if no hyperparameter optimization is performed.
# test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

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

print("Successfully ran ex2_hyperparameter_optimization.py.")
print("Parameters used for this experiment:")
test_parameters.show()
