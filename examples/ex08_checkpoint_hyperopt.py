import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex08_checkpoint_hyperopt.py: Shows how a hyperparameter optimization run can 
be paused and resumed. Delete all ex08_*.pkl and ex08_*.pth prior to execution.
Afterwards, execute this script twice to see how MALA progresses from a 
checkpoint. As the number of trials cannot be divided by the number
of epochs after which a checkpoint is created without residual, this will 
lead to MALA performing the missing trials again.
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
    test_parameters.running.max_number_epochs = 10
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    # The study will be 9 trials long, with checkpoints after the 5th
    # trial. This will result in the first checkpoint still having 4
    # trials left.
    test_parameters.hyperparameters.n_trials = 9
    test_parameters.hyperparameters.hyper_opt_method = "optuna"
    test_parameters.hyperparameters.checkpoints_each_trial = 5
    test_parameters.hyperparameters.checkpoint_name = "ex08"

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
    # After first performing the study, we load it from the last checkpoint
    # and run the remaining trials.
    ####################

    test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)

    # Learning rate will be optimized.
    test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                         0.0000001, 0.01)

    # Number of neurons per layer will be optimized.
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
    test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)

    # Choices for activation function at each layer will be optimized.
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_00",
                                         choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_01",
                                         choices=["ReLU", "Sigmoid"])
    test_hp_optimizer.add_hyperparameter("categorical", "layer_activation_02",
                                         choices=["ReLU", "Sigmoid"])

    return test_parameters, data_handler, test_hp_optimizer


def run_example08():
    if mala.HyperOptOptuna.checkpoint_exists("ex08"):
        parameters, datahandler, hyperoptimizer = \
            mala.HyperOptOptuna.resume_checkpoint(
                "ex08")
        printout("Starting resumed hyperparameter optimization.")
    else:
        parameters, datahandler, hyperoptimizer = initial_setup()
        printout("Starting original hyperparameter optimization.")

    # Perform hyperparameter optimization.
    hyperoptimizer.perform_study()

    ####################
    # RESULTS.
    # Print the used parameters.
    ####################

    printout("Parameters used for this experiment:")
    parameters.show()
    return True


if __name__ == "__main__":
    if run_example08():
        printout("Successfully ran ex08_checkpoint_hyperopt.")
    else:
        raise Exception("Ran ex08_checkpoint_hyperopt but something was off."
                        " If you haven't changed any parameters in "
                        "the example, there might be a problem with your"
                        " installation.")
