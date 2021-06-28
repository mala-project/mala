import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

"""
ex04_hyperparameter_optimization.py: Shows how a hyperparameter 
optimization can be done using this framework. There are multiple 
hyperparameter optimizers available in this framework. This example focusses
on the most universal one - optuna.  
"""


def run_example04(desired_loss_improvement_factor=1):
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
    test_parameters.running.max_number_epochs = 5
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    test_parameters.hyperparameters.n_trials = 20
    test_parameters.hyperparameters.hyper_opt_method = "oat"

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

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    test_hp_optimizer.perform_study()
    test_hp_optimizer.set_optimal_parameters()
    printout("Hyperparameter optimization: DONE.")

    ####################
    # TRAINING
    # Train with these new parameters.
    ####################

    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")
    test_trainer.train_network()
    printout("Training: DONE.")

    ####################
    # RESULTS.
    # Print the used parameters and check whether the loss decreased enough.
    ####################

    printout("Parameters used for this experiment:")
    test_parameters.show()

    if desired_loss_improvement_factor*test_trainer.initial_test_loss < \
            test_trainer.final_test_loss:
        return False
    else:
        return True


if __name__ == "__main__":
    if run_example04():
        printout("Successfully ran ex04_hyperparameter_optimization.py.")
    else:
        raise Exception("Ran ex04_hyperparameter_optimization but something "
                        "was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with "
                        "your installation.")
