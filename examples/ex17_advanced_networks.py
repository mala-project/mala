import mala
from mala import printout
import os
from data_repo_path import get_data_repo_path
data_path = os.path.join(get_data_repo_path(), "Al36/")

"""
ex17_advanced_networks.py: Shows how advanced network models such as 
lstm, gru and transformer can be used for prediction.
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
test_parameters.running.max_number_epochs = 30
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"
test_parameters.use_gpu= True
####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                          "Al_debug_2k_nr0.out.npy", data_path,
                          add_snapshot_as="tr", output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                          "Al_debug_2k_nr1.out.npy", data_path,
                          add_snapshot_as="va", output_units="1/Ry")
data_handler.prepare_data()
printout("Read data: DONE.")



####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################
test_parameters.network.nn_type = "transformer"
# test_parameters.network.nn_type = "lstm"
# test_parameters.network.nn_type = "gru"

test_parameters.network.num_hidden_layers = 1
test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),100,
                                       data_handler.get_output_dimension()]

if test_parameters.network.nn_type == "lstm" or \
        test_parameters.network.nn_type == "gru":
    test_parameters.network.no_hidden_state = False
    test_parameters.network.bidirection = False
elif test_parameters.network.nn_type == "transformer":
    test_parameters.network.dropout = 0.2

    # This value has to be a divisor of the input dimension. If this is
    # for some reason incorrectly provided by the user, MALA will determine
    # the next highest divisor.
    test_parameters.network.num_heads = 7

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
