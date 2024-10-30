import os

import mala

from mala.datahandling.data_repo import data_path

"""
This example shows how a neural network can be trained on material
data using this framework. It uses preprocessed data, that is read in
from *.npy files.
"""

####################
# 1. PARAMETERS
# The first step of each MALA workflow is to define a parameters object and
# select the necessary parameters for the application one wants to look into.
####################

parameters = mala.Parameters()
# Specify the data scaling. For regular bispectrum and LDOS data,
# these have proven successful.
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
# Specify the used activation function.
parameters.network.layer_activations = ["ReLU"]
# Specify the training parameters.
# These may be determined via hyperparameter tuning.
parameters.running.max_number_epochs = 100
parameters.running.mini_batch_size = 40
parameters.running.learning_rate = 0.00001
parameters.running.optimizer = "Adam"
# These parameters characterize how the LDOS and bispectrum descriptors
# were calculated. They are _technically_ not needed to train a simple
# network. However, it is useful to define them prior to training. Then,
# when using the network later in production, all required parameters are
# already set.
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

parameters.descriptors.descriptor_type = "Bispectrum"
parameters.descriptors.bispectrum_twojmax = 10
parameters.descriptors.bispectrum_cutoff = 4.67637

####################
# 2. DATA
# Data has to be added to the MALA workflow. The central object for this
# is the DataHandler class, which takes care of all data needs. After data
# has been added, it is loaded and scaled with the prepare_data function.
####################

data_handler = mala.DataHandler(parameters)
# Add a snapshot we want to use in to the list.
data_handler.add_snapshot(
    "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path, "tr"
)
data_handler.add_snapshot(
    "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path, "va"
)
data_handler.prepare_data()

####################
# 3. NETWORK SETUP
# Now we can set up the NN to be used in the ML-DFT model. The layer_sizes
# list determines the number of neurons in the NN. It can be specified before
# loading data, but it is recommended to do that afterwards, since then
# the input_dimension and output_dimension properties of the data handling
# class can be used to correctly define input and output layer of the NN.
####################

parameters.network.layer_sizes = [
    data_handler.input_dimension,
    100,
    data_handler.output_dimension,
]
test_network = mala.Network(parameters)

####################
# 4. TRAINING THE NETWORK
# Finally, the network can be trained. Afterwards, it can easily be saved
# into a .zip archive for inference. It is recommended to load a file
# containing additional calculation data (e.g., from the QE calculations
# with which the LDOS data was created) so that things like simulated
# temperature, information about the pseudopotential, etc. are stored along-
# side the model. This makes inference easier.
####################

test_trainer = mala.Trainer(parameters, test_network, data_handler)
test_trainer.train_network()
additional_calculation_data = os.path.join(data_path, "Be_snapshot0.out")
test_trainer.save_run(
    "Be_model", additional_calculation_data=additional_calculation_data
)
