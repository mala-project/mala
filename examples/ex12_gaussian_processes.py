import gpytorch
import torch

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
data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################
print(data_handler.training_data_inputs.size())
data_handler.training_data_inputs = data_handler.training_data_inputs.transpose(0, 1)
data_handler.training_data_inputs = data_handler.training_data_inputs.reshape(2, 100, 1)
data_handler.training_data_outputs = data_handler.training_data_outputs.transpose(0, 1)
# data_handler.training_data_outputs = data_handler.training_data_outputs.reshape(1, 100, 1)
print(data_handler.training_data_inputs.size())
model = mala.GaussianProcesses(params, data_handler)
model.train()
model.likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
training_iter = 50
for i in range(training_iter):
    model.train()
    model.likelihood.train()
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(data_handler.training_data_inputs)
    # Calc loss and backprop gradients
    loss = -mll(output, data_handler.training_data_outputs).sum()
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model.eval()
    model.likelihood.eval()
    test = model.likelihood(model(data_handler.validation_data_inputs.transpose(0, 1).reshape(2, 100, 1)))


# # Setup network and trainer.
# model = mala.GaussianProcesses(params, data_handler)
# trainer = mala.Trainer(params, model, data_handler)
# printout("Network setup: DONE.")
#
# ####################
# # TRAINING
# # Train the network.
# ####################
#
# printout("Starting training.")
# trainer.train_model()
# printout("Training: DONE.")
#
# tester = mala.Tester(params, model, data_handler)
#
# ####################
# # TESTING
# # Pass the first test set snapshot (the test snapshot).
# ####################
#
# actual_density, predicted_density = tester.test_snapshot(0)
# # First test snapshot --> 4th in total
# data_handler.target_calculator.read_additional_calculation_data("qe.out", data_handler.get_snapshot_calculation_output(4))
# actual_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(actual_density)
# predicted_number_of_electrons = data_handler.target_calculator.get_number_of_electrons(predicted_density)
# printout(actual_number_of_electrons, predicted_number_of_electrons)
