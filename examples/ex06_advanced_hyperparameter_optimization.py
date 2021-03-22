import mala
from mala import printout
import numpy as np
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"

"""
ex06_advanced_hyperparameter_optimization.py: Shows how recent developments 
in hyperparameter optimization techniques can be used 
(OAT / training-free NAS).
"""

printout("Welcome to MALA.")
printout("Running ex06_advanced_hyperparameter_optimization.py")


def optimize_hyperparameters(hyper_optimizer, input_creator_notraining="oat",
                             last_optuna_study=None):
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

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 20
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
    # HYPERPARAMETER OPTIMIZATION
    # In order to perform a hyperparameter optimization,
    # one has to simply create a hyperparameter optimizer
    # and let it perform a "study".
    # Before such a study can be done, one has to add all the parameters
    # of interest.
    ####################

    test_hp_optimizer = mala.HyperOptInterface(test_parameters, data_handler)
    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                           100, 100,
                                           data_handler.get_output_dimension()]

    # Add hyperparameters we want to have optimized to the list.
    # If we do a notraining run currently we need to create an input array
    # using one of the other two possible hyperparameter optimizers.
    tmp_hp_optimizer = None
    if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
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
    elif hyper_optimizer == "notraining":
        tmp_parameters = test_parameters
        if input_creator_notraining == "optuna" and last_optuna_study is None:
            input_creator_notraining = "oat"
        tmp_parameters.hyperparameters.hyper_opt_method = \
            input_creator_notraining
        tmp_hp_optimizer = mala.HyperOptInterface(tmp_parameters, data_handler)
        tmp_hp_optimizer.add_hyperparameter("categorical", "trainingtype",
                                            choices=["Adam", "SGD"])
        tmp_hp_optimizer.add_hyperparameter("categorical",
                                            "layer_activation_00",
                                            choices=["ReLU", "Sigmoid"])
        tmp_hp_optimizer.add_hyperparameter("categorical",
                                            "layer_activation_01",
                                            choices=["ReLU", "Sigmoid"])
        tmp_hp_optimizer.add_hyperparameter("categorical",
                                            "layer_activation_02",
                                            choices=["ReLU", "Sigmoid"])

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
        test_hp_optimizer.perform_study()
        if hyper_optimizer == "optuna":
            last_optuna_study = test_hp_optimizer.get_trials_from_study()
    elif hyper_optimizer == "notraining":
        if input_creator_notraining == "optuna":
            test_hp_optimizer.perform_study(trial_list=last_optuna_study)
        else:
            test_hp_optimizer.perform_study(trial_list=
                                            tmp_hp_optimizer.orthogonal_arr)
    test_hp_optimizer.set_optimal_parameters()
    printout("Hyperparameter optimization: DONE.")

    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")
    test_trainer.train_network()
    printout("Training: DONE.")

    printout("Parameters used for this experiment:")
    test_parameters.show()
    if hyper_optimizer == "optuna":
        return test_trainer.final_test_loss, last_optuna_study
    else:
        return test_trainer.final_test_loss


def run_example06(desired_std=0.1):
    results = []
    result, last_study = optimize_hyperparameters("optuna")
    results.append(result)
    results.append(optimize_hyperparameters("oat"))
    results.append(optimize_hyperparameters("notraining",
                                            input_creator_notraining="oat"))
    results.append(optimize_hyperparameters("notraining",
                                            input_creator_notraining="optuna",
                                            last_optuna_study=last_study))
    if np.std(results) < desired_std:
        return True
    else:
        printout(results)
        printout(np.std(results))
        return False


if __name__ == "__main__":
    if run_example06():
        printout("Successfully ran ex06_advanced_hyperparameter_optimization."
                 "py.")
    else:
        raise Exception("Ran ex06_advanced_hyperparameter_optimization but "
                        "something was off. If you haven't changed any "
                        "parameters in the example, there might be a problem "
                        "with your installation.")
