from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.network.OAT_hyperparameter_optimiser import OAT_HyperOpt
from fesl.network.OAT_parameter import OA_Parameter

"""
ex09_hyperparameter_optimization.py: Shows how a hyperparameter optimization can be done using this framework.
"""

print("Welcome to FESL.")
print("Running ex02_hyperparameter_optimization.py")

####################
# PARAMETERS
# All parameters are handled from a central parameters class that contains subclasses.
####################
test_parameters = Parameters()
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


####################
# DATA
# Read data into RAM.
# We have to specify the directories we want to read the snapshots from.
# The Handlerinterface will also return input and output scaler objects. These are used internally to scale
# the data. The objects can be used after successful training for inference or plotting.
####################
data_handler = HandlerInterface(test_parameters)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/", output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", "./data/", "Al_debug_2k_nr1.out.npy", "./data/", output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")
data_handler.load_data()
data_handler.prepare_data()
print("Read data: DONE.")

####################
# HYPERPARAMETER OPTIMIZATION
# In order to perform a hyperparameter optimization,
# one has to simply create a hyperparameter optimizer
# and let it perform a "study".
# This class is nothing more than a wrapper to optuna.
# Via the hyperparameter_list one can add hyperparameters to be optimized.
# Please not that you have to give specifications by hand for hyperparameters you do not want to optimize.
####################
test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, 100, data_handler.get_output_dimension()]

# Add hyperparameters we want to have optimized to the list.
# test_parameters.hyperparameters.hlist.append(OptunaParameter("float", "learning_rate", 0.0000001, 0.01))
test_parameters.hyperparameters.hlist.append(
    OA_Parameter("categorical", "trainingtype", choices=["Adam", "SGD"]))

# We want to optimize the choices for the activation functions at EACH layer. We can do this in a similarly manner.
test_parameters.hyperparameters.hlist.append(
    OA_Parameter("categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]))
test_parameters.hyperparameters.hlist.append(
    OA_Parameter("categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]))
test_parameters.hyperparameters.hlist.append(
    OA_Parameter("categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]))

# Perform hyperparameter optimization.
print("Starting Hyperparameter optimization.")
test_hp_optimizer = OAT_HyperOpt(test_parameters)
test_hp_optimizer.perform_study(data_handler)
test_hp_optimizer.set_optimal_parameters()
print("Hyperparameter optimization: DONE.")

print("Successfully ran ex09_OAT_hyperparameter_optimization.py.")
print("Parameters used for this experiment:")
test_parameters.show()