import fesl
from fesl import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"

"""
ex02_hyperparameter_optimization.py: Shows how a hyperparameter optimization can be done using this framework.
"""
def run_example02(desired_loss_improvement_factor=1):
    printout("Welcome to FESL.")
    printout("Running ex02_hyperparameter_optimization.py")

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = fesl.Parameters()
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.hyperparameters.n_trials = 20
    test_parameters.comment = "Test run of FESL."


    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################
    data_handler = fesl.DataHandler(test_parameters)

    # Add all the snapshots we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path, "Al_debug_2k_nr0.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path, "Al_debug_2k_nr1.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path, "Al_debug_2k_nr2.out.npy", data_path, output_units="1/Ry")
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

    test_hp_optimizer = fesl.HyperOptInterface(test_parameters)
    # Add hyperparameters we want to have optimized to the list.
    test_hp_optimizer.add_hyperparameter("float", "learning_rate", 0.0000001, 0.01)

    # We want a network with two layers, so we can just add two parameters for the number for neurons per layer.
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)

    # We also want to optimize the choices for the activation functions at EACH layer. We can do this in a similarly manner.
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"])

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    test_hp_optimizer.perform_study(data_handler)
    test_hp_optimizer.set_optimal_parameters()
    printout("Hyperparameter optimization: DONE.")

    ####################
    # TRAINING
    # Train wit these new parameters.
    ####################

    test_network = fesl.Network(test_parameters)
    test_trainer = fesl.Trainer(test_parameters)
    printout("Network setup: DONE.")
    test_trainer.train_network(test_network, data_handler)
    printout("Training: DONE.")

    printout("Parameters used for this experiment:")
    test_parameters.show()

    if desired_loss_improvement_factor*test_trainer.initial_test_loss < test_trainer.final_test_loss:
        return False
    else:
        return True


if __name__ == "__main__":
    if run_example02():
        printout("Successfully ran ex02_hyperparameter_optimization.py.")
    else:
        raise Exception("Ran ex02_hyperparameter_optimization but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")

