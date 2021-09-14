import mala
from mala import printout
data_path = "/home/fiedlerl/jobs/qe/Be2/data_for_qgp/"

"""
ex12_hyperparameter_optimization.py: Shows how Gaussian processes can be used
to learn the electronic density with MALA. Backend is GPytorch.
"""

params = mala.Parameters()

# Specify the data scaling.
params.data.input_rescaling_type = "feature-wise-standard"
params.data.output_rescaling_type = "normal"

# Specify the used activation function.
params.model.loss_function_type = "gaussian_likelihood"

# Specify the training parameters.
params.running.max_number_epochs = 50
params.running.mini_batch_size = 1
params.running.learning_rate = 0.1
params.running.trainingtype = "Adam"
params.targets.target_type = "Density"
params.debug.grid_dimensions = [10, 10, 1]
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
                          "snapshot1.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.add_snapshot("snapshot2.in.npy", inputs_folder,
                          "snapshot2.out.npy", outputs_folder, add_snapshot_as="va", output_units="None")
data_handler.prepare_data(transpose_data=True)
printout("Read data: DONE.")

####################
# MODEL SETUP
# Set up the model and trainer we want to use.
####################

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


####################
# TESTING
# Pass the first test set snapshot (the test snapshot).
####################

tester = mala.Tester(params, model, data_handler)
actual_density, predicted_density = tester.test_snapshot(0)
# First test snapshot --> 4th in total
data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(4))
actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
printout(actual_number_of_electrons, predicted_number_of_electrons)
