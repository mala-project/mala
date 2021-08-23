
import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = "/home/fiedlerl/data/mala_data_repo_FULL/Al36_density/"

"""
ex04_hyperparameter_optimization.py: Shows how a hyperparameter
optimization can be done using this framework. There are multiple
hyperparameter optimizers available in this framework. This example focusses
on the most universal one - optuna.
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
test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

# Specify the data scaling.
test_parameters.data.input_rescaling_type = "feature-wise-standard"
test_parameters.data.output_rescaling_type = "normal"
test_parameters.model.type = "dummy"
test_parameters.running.use_shuffling_for_samplers = False

# Specify the training parameters.
test_parameters.running.max_number_epochs = 1
test_parameters.running.mini_batch_size = 40
test_parameters.running.learning_rate = 0.001
test_parameters.running.trainingtype = "Adam"

# Specify the number of trials, the hyperparameter optimizer should run
# and the type of hyperparameter.
test_parameters.hyperparameters.n_trials = 20
test_parameters.hyperparameters.hyper_opt_method = "optuna"

####################
# DATA
# Add and prepare snapshots for training.
####################
data_handler = mala.DataHandler(test_parameters)

# Add all the snapshots we want to use in to the list.
data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                          "Al_debug_2k_nr0.out.npy", data_path,
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                          "Al_debug_2k_nr1.out.npy", data_path,
                          output_units="1/Ry")
data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                          "Al_debug_2k_nr2.out.npy", data_path,
                          output_units="1/Ry")
data_handler.prepare_data()


printout("Read data: DONE.")

####################
# TRAINING
# Train a model.
####################

model = mala.DummyModel(test_parameters)
test_trainer = mala.Trainer(test_parameters, model, data_handler)
test_trainer.train_network()

####################
# TESTING
# Evaluate model performance.
####################

tester = mala.Tester(test_parameters, model, data_handler)
actual_density, predicted_density = tester.test_snapshot(2)

# For this model, the target calculator handles density data.
density_calculator = data_handler.target_calculator
density_calculator.read_additional_calculation_data("qe.out",
                                                    data_path +
                                                    "Al.pw.scf.out")

# Calculate the number of electrons.
nr_electrons_predicted = density_calculator. \
    get_number_of_electrons(predicted_density)
nr_electrons_actual = density_calculator.get_number_of_electrons(actual_density)
printout("Number of electrons (actual, predicted, error)[eV]",
         nr_electrons_actual, nr_electrons_predicted,
         nr_electrons_predicted - nr_electrons_actual)



