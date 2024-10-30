import os

import mala

from mala.datahandling.data_repo import data_path

"""
Shows how a hyperparameter optimization can be done using this framework. There
are multiple hyperparameter optimizers available in this framework. This example
focusses on the most universal one - optuna.
"""


####################
# 1. PARAMETERS
# Set the parameters, similar to it is done for a training procedure.
# Additionally to the parameters used there we also have to specify the number
# of trials to be performed by optuna.
####################
parameters = mala.Parameters()
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
parameters.running.max_number_epochs = 20
parameters.running.mini_batch_size = 40
parameters.running.optimizer = "Adam"
parameters.hyperparameters.n_trials = 20

####################
# 2. DATA
# Data is added in the same way it is done for training a model.
####################
data_handler = mala.DataHandler(parameters)
data_handler.add_snapshot(
    "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path, "tr"
)
data_handler.add_snapshot(
    "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path, "va"
)
data_handler.prepare_data()

####################
# 3. ADDING HYPERPARAMETERS.
# Now we have to add the hyperparameters we want to optimize.
# This is done via an interface to the HyperOpt class, and for each
# parameter we have to either specify a list of potential choices or a range
# to choose from.
# Here we will optimize the learning rate and the number of neurons per layer,
# as well as the activation function.
####################

hyperoptimizer = mala.HyperOpt(parameters, data_handler)
hyperoptimizer.add_hyperparameter(
    "categorical", "learning_rate", choices=[0.005, 0.01, 0.015]
)
hyperoptimizer.add_hyperparameter(
    "categorical", "ff_neurons_layer_00", choices=[32, 64, 96]
)
hyperoptimizer.add_hyperparameter(
    "categorical", "ff_neurons_layer_01", choices=[32, 64, 96]
)
hyperoptimizer.add_hyperparameter(
    "categorical",
    "layer_activation_00",
    choices=["ReLU", "Sigmoid", "LeakyReLU"],
)

####################
# 4. PERFORMING THE HYPERPARAMETER STUDY.
# Finally, we can perform the actual hyperparameter study. After the study
# has finished, we can save the optimal parameters to the linked parameters
# object, and e.g., save them to file. They can then be loaded for building the
# the next model with params = mala.Parameters.load_from_file(...)
####################
hyperoptimizer.perform_study()
hyperoptimizer.set_optimal_parameters()
hyperoptimizer.params.save("optimal_params")
