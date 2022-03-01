import mala
from mala import printout
from data_repo_path import data_repo_path
data_path = data_repo_path+"Al36/"
# data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")


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
# done on a "by snapshot" basis. Specify how this is
# done by providing a list containing entries of the form
# "tr", "va" and "te".
# test_parameters.data.data_splitting_type = "by_snapshot"

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"

# Specify the used activation function.
test_parameters.network.layer_activations = ["ReLU"]

# Specify the training parameters.
test_parameters.running.max_number_epochs = 300
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.00001
test_parameters.running.trainingtype = "Adam"
test_parameters.use_gpu= True
####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(test_parameters)

# # Add a snapshot we want to use in to the list.


data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                          "Al_debug_2k_nr0.out.npy", data_path, "tr",
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                          "Al_debug_2k_nr1.out.npy", data_path, "va",
                          output_units="1/Ry")
####################
#BE2 data
####################

# data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
#                           "Be_snapshot1.out.npy", data_path,
#                           output_units="1/Ry")##calculation outpu
# data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
#                           "Be_snapshot2.out.npy", data_path,
#                           output_units="1/Ry")

####################
#Big data
####################

# snap_path= "/bigdata/casus/wdm/Al/298K/N256/snap"
# ldos_path= "/bigdata/casus/wdm/Al/298K/N256/ldos"

# training_number = 1
# validation_number = 1

# for i in range(0,training_number):
#     data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot"+str(i)+".npy", snap_path,
#                               "Al_ldos_200x200x200grid_250elvls_snapshot"+str(i)+".npy", ldos_path,
#                               add_snapshot_as="tr", output_units="1/Ry") 

# for i in range(training_number,training_number+validation_number):
#     data_handler.add_snapshot("Al_fp_200x200x200grid_94comps_snapshot"+str(i)+".npy", snap_path,
#                               "Al_ldos_200x200x200grid_250elvls_snapshot"+str(i)+".npy", ldos_path,
#                               add_snapshot_as="va", output_units="1/Ry")

data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################
# test_parameters.network.nn_type = "transformer"
# test_parameters.network.nn_type = "lstm"
test_parameters.network.nn_type = "gru"
# test_parameters.network.nn_type = "feed-forward"

test_parameters.network.num_hidden_layers = 1
test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),100,
                                       data_handler.get_output_dimension()]

if test_parameters.network.nn_type == "lstm" or test_parameters.network.nn_type == "gru":
    test_parameters.network.no_hidden_state = False
    test_parameters.network.bidirection = False
elif test_parameters.network.nn_type == "transformer":
    test_parameters.network.dropout = 0.2
    test_parameters.network.num_heads= 7 #must be a divisor of input dimension

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

# printout("Parameters used for this experiment:")
# test_parameters.show()

# Specify the correct LDOS parameters.
test_parameters.targets.target_type = "LDOS"
test_parameters.targets.ldos_gridsize = 250
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10

# Inference should ALWAYS be done with lazy loading activated, even if
# training was not.
test_parameters.data.use_lazy_loading = True

test_parameters.data.data_splitting_snapshots = ["te"]
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                          "Al_debug_2k_nr2.out.npy", data_path,
                          output_units="1/Ry")
data_handler.prepare_data(reparametrize_scaler=False)

# The Tester class is the testing analogon to the training class.
tester = mala.Tester(test_parameters, test_network, data_handler)

# Get the results for the first (and only= snapshot.
actual_ldos, predicted_ldos = tester.test_snapshot(0)

# We will use the LDOS calculator to do some preprocessing.
ldos_calculator = data_handler.target_calculator
ldos_calculator.read_additional_calculation_data("qe.out", data_path+"Al.pw.scf.out")

# Calculate the Band energy.
band_energy_predicted = ldos_calculator.get_band_energy(predicted_ldos)
band_energy_actual = ldos_calculator.get_band_energy(actual_ldos)
printout("Band energy (actual, predicted, error)[eV]", band_energy_actual,
            band_energy_predicted, band_energy_predicted-band_energy_actual)

# Calculate the number of electrons.
nr_electrons_predicted = ldos_calculator.\
    get_number_of_electrons(predicted_ldos)
nr_electrons_actual = ldos_calculator.get_number_of_electrons(actual_ldos)
printout("Number of electrons (actual, predicted, error)[eV]",
            nr_electrons_actual, nr_electrons_predicted,
            nr_electrons_predicted-nr_electrons_actual)