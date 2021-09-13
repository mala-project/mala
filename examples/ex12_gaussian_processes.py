import mala
from mala import printout
data_path = "/home/fiedlerl/jobs/qe/Be2/data_for_qgp/"

"""
ex04_hyperparameter_optimization.py: Shows how a hyperparameter
optimization can be done using this framework. There are multiple
hyperparameter optimizers available in this framework. This example focusses
on the most universal one - optuna.
"""


params = mala.Parameters()

# Specify the data scaling.
params.data.input_rescaling_type = "feature-wise-standard"
params.data.output_rescaling_type = "normal"

# Specify the used activation function.
params.model.loss_function_type = "gaussian_likelihood"

# Specify the training parameters.
params.running.max_number_epochs = 100
params.running.mini_batch_size = 40
params.running.learning_rate = 0.0001
params.running.trainingtype = "Adam"
params.targets.target_type = "Density"

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(params)
inputs_folder = data_path+"inputs_snap/"
outputs_folder = data_path+"outputs_density/"
additional_folder = data_path+"additional_info_qeouts/"
# Add a snapshot we want to use in to the list.
data_handler.add_snapshot("snapshot0.in.npy", inputs_folder,
                          "snapshot0.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot1.in.npy", inputs_folder,
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="tr", output_units="None")
data_handler.add_snapshot("snapshot3.in.npy", inputs_folder,
                          "snapshot3.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("snapshot4.in.npy", inputs_folder,
                          "snapshot4.out.npy", outputs_folder, add_snapshot_as="te", output_units="None",
                          calculation_output_file=additional_folder+"snapshot4.out")
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

params.model.layer_sizes = [data_handler.get_input_dimension(),
                                       100,
                                       data_handler.get_output_dimension()]

# Setup network and trainer.
model = mala.GaussianProcesses(params, data_handler)
trainer = mala.Trainer(params, model, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network.
####################

printout("Starting training.")
trainer.train_model()
printout("Training: DONE.")

tester = mala.Tester(params, model, data_handler)

####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

actual_density, predicted_density = tester.test_snapshot(0)
# First test snapshot --> 4th in total
data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(4))
actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
printout(actual_number_of_electrons, predicted_number_of_electrons)
