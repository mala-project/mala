import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Be2/densities_gp/"

"""
ex01_run_singleshot.py: Shows how a neural models can be trained on material 
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
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
test_parameters.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the used activation function.
test_parameters.model.layer_activations = ["ReLU"]

# Specify the training parameters.
test_parameters.running.max_number_epochs = 20
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"
test_parameters.targets.target_type = "Density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
inputs_folder = data_path+"inputs_snap/"
outputs_folder = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                          "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="te",
                          output_units="None", calculation_output_file=additional_folder+"snapshot2.out")
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the models and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

test_parameters.model.layer_sizes = [data_handler.get_input_dimension(),
                                     100,
                                     data_handler.get_output_dimension()]

# Setup models and trainer.
test_network = mala.Network(test_parameters)
test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the models.
####################

printout("Starting training.")
test_trainer.train_model()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
test_parameters.show()
