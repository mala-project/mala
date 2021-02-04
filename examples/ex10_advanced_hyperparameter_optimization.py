from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.network.hyper_opt_interface import HyperOptInterface
from fesl.network.network import Network
from fesl.network.trainer import Trainer
import numpy as np

"""
ex10_advanced_hyperparameter_optimization.py: Shows how recent developments in hyperparameter optimization techniques can be used
in FESL (OAT / training-free NAS).
"""

printout("Welcome to FESL.")
printout("Running ex02_hyperparameter_optimization.py")


def optimize_hyperparameters(hyper_optimizer, input_creator_notraining="oat", last_optuna_study = None):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = Parameters()
    test_parameters.data.datatype_in = "*.npy"
    test_parameters.data.datatype_out = "*.npy"
    test_parameters.data.data_splitting_percent = [80, 10, 10]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.training.max_number_epochs = 20
    test_parameters.training.mini_batch_size = 40
    test_parameters.training.learning_rate = 0.00001
    test_parameters.training.trainingtype = "Adam"
    test_parameters.hyperparameters.n_trials = 8
    test_parameters.comment = "Test run of FESL."
    test_parameters.hyperparameters.hyper_opt_method = hyper_optimizer


    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################
    data_handler = HandlerInterface(test_parameters)

    # Add all the snapshots we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", "./data/", "Al_debug_2k_nr1.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")
    data_handler.load_data()
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # HYPERPARAMETER OPTIMIZATION
    # In order to perform a hyperparameter optimization,
    # one has to simply create a hyperparameter optimizer
    # and let it perform a "study".
    # This class is nothing more than a wrapper to optuna.
    # Via the hyperparameter_list one can add hyperparameters to be optimized.
    # Please not that you have to give specifications by hand for hyperparameters you do not want to optimize.
    ####################

    test_hp_optimizer = HyperOptInterface(test_parameters)
    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, 100, data_handler.get_output_dimension()]


    # Add hyperparameters we want to have optimized to the list.
    # If we do a notraining run currently we need to create an input array using one of the other two possible
    # hyperparameter optimizers.
    if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
        test_hp_optimizer.add_hyperparameter("categorical", "trainingtype", choices=["Adam", "SGD"])
        test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"])
    elif hyper_optimizer == "notraining":
        tmp_parameters=test_parameters
        if input_creator_notraining == "optuna" and last_optuna_study is None:
            input_creator_notraining = "oat"
        tmp_parameters.hyperparameters.hyper_opt_method = input_creator_notraining
        tmp_hp_optimizer = HyperOptInterface(tmp_parameters)
        tmp_hp_optimizer.add_hyperparameter("categorical", "trainingtype", choices=["Adam", "SGD"])
        tmp_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"])
        tmp_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"])
        tmp_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"])

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
        test_hp_optimizer.perform_study(data_handler)
        if hyper_optimizer == "optuna":
            last_optuna_study = test_hp_optimizer.study.get_trials()
    elif hyper_optimizer == "notraining":
        if input_creator_notraining == "optuna":
            test_hp_optimizer.perform_study(data_handler,trial_list=last_optuna_study)
        else:
            test_hp_optimizer.perform_study(data_handler,trial_list=tmp_hp_optimizer.orthogonal_arr)
    test_hp_optimizer.set_optimal_parameters()
    printout("Hyperparameter optimization: DONE.")

    test_network = Network(test_parameters)
    test_trainer = Trainer(test_parameters)
    printout("Network setup: DONE.")
    test_trainer.train_network(test_network, data_handler)
    printout("Training: DONE.")

    printout("Parameters used for this experiment:")
    test_parameters.show()
    if hyper_optimizer == "optuna":
        return test_trainer.final_test_loss, last_optuna_study
    else:
        return test_trainer.final_test_loss


def run_example10(desired_std = 0.1):
    results = []
    result, last_study = optimize_hyperparameters("optuna")
    results.append(result)
    results.append(optimize_hyperparameters("oat"))
    results.append(optimize_hyperparameters("notraining", input_creator_notraining="oat"))
    results.append(optimize_hyperparameters("notraining", input_creator_notraining="optuna", last_optuna_study=last_study))
    if np.std(results) < desired_std:
        return True
    else:
        printout(results)
        printout(np.std(results))
        return False

if __name__ == "__main__":
    if run_example10():
        printout("Successfully ran ex10_advanced_hyperparameter_optimization.py.")
    else:
        raise Exception("Ran ex10_advanced_hyperparameter_optimization but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")
