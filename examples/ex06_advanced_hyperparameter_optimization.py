import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex06_advanced_hyperparameter_optimization.py: Shows how recent developments 
in hyperparameter optimization techniques can be used 
(OAT / training-free NAS).


REQUIRES OAPACKAGE.
"""

printout("Welcome to MALA.")
printout("Running ex06_advanced_hyperparameter_optimization.py")


def optimize_hyperparameters(hyper_optimizer):
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
    test_parameters.running.max_number_epochs = 10
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    test_parameters.hyperparameters.n_trials = 8
    test_parameters.hyperparameters.hyper_opt_method = hyper_optimizer

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
    test_parameters.network.layer_sizes = [data_handler.input_dimension,
                                           100, 100,
                                           data_handler.output_dimension]

    # Add hyperparameters we want to have optimized to the list.
    # If we do a NASWOT run currently we need to create an input array
    # using one of the other two possible hyperparameter optimizers.
    test_hp_optimizer.add_hyperparameter("categorical", "trainingtype",
                                         choices=["Adam", "SGD"])
    test_hp_optimizer.add_hyperparameter("categorical",
                                         "layer_activation_00",
                                         choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical",
                                         "layer_activation_01",
                                         choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical",
                                         "layer_activation_02",
                                         choices=["ReLU", "Sigmoid"])

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    test_hp_optimizer.perform_study()
    test_hp_optimizer.set_optimal_parameters()
    printout("Hyperparameter optimization: DONE.")

    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")
    test_trainer.train_network()
    printout("Training: DONE.")

    printout("Parameters used for this experiment:")
    test_parameters.show()
    return test_trainer.final_test_loss


results = [optimize_hyperparameters("optuna"), optimize_hyperparameters("oat"),
           optimize_hyperparameters("naswot")]
print(results)
