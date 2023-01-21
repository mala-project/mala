import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex04_hyperparameter_optimization.py: Shows how a hyperparameter 
optimization can be done using this framework. There are multiple 
hyperparameter optimizers available in this framework. This example focusses
on the most universal one - optuna.  
"""


def initial_setup():
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

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    test_parameters.hyperparameters.n_trials = 20
    test_parameters.hyperparameters.hyper_opt_method = "optuna"
    test_parameters.hyperparameters.number_training_per_trial = 1
    test_parameters.running.verbosity = False
    test_parameters.hyperparameters.checkpoint_name = "ex04"
    test_parameters.hyperparameters.checkpoints_each_trial = -1

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    data_handler = mala.DataHandler(test_parameters)

    # Add all the snapshots we want to use in to the list.
    data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                              "Be_snapshot0.out.npy", data_path, "tr")
    data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                              "Be_snapshot1.out.npy", data_path, "va")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path, "te")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # HYPERPARAMETER OPTIMIZATION
    # In order to perform a hyperparameter optimization,
    # one has to simply create a hyperparameter optimizer
    # and let it perform a "study".
    # Before such a study can be done, one has to add all the parameters
    # of interest.
    ####################

    test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)

    # Learning rate will be optimized.
    test_hp_optimizer.add_hyperparameter("categorical", "learning_rate",
                                         choices=[0.005, 0.01, 0.015])

    # Number of neurons per layer will be optimized.
    test_hp_optimizer.add_hyperparameter(
        "categorical", "ff_neurons_layer_00", choices=[32, 64, 96])
    test_hp_optimizer.add_hyperparameter(
        "categorical", "ff_neurons_layer_01", choices=[32, 64, 96])

    # Choices for activation function at each layer will be optimized.
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00",
                                         choices=["ReLU", "Sigmoid", "LeakyReLU"])
    return test_parameters, data_handler, test_hp_optimizer


if mala.HyperOptOptuna.checkpoint_exists("ex04"):
    parameters, datahandler, hyperoptimizer = \
        mala.HyperOptOptuna.resume_checkpoint(
            "ex04")
    printout("Starting resumed hyperparameter optimization.")
else:
    parameters, datahandler, hyperoptimizer = initial_setup()
    printout("Starting original hyperparameter optimization.")


# Perform hyperparameter optimization.
printout("Starting Hyperparameter optimization.")
hyperoptimizer.perform_study()
hyperoptimizer.set_optimal_parameters()
printout("Hyperparameter optimization: DONE.")

####################
# TRAINING
# Train with these new parameters.
####################

test_network = mala.Network(parameters)
test_trainer = mala.Trainer(parameters, test_network, datahandler)
printout("Network setup: DONE.")
test_trainer.train_network()
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
parameters.show()
