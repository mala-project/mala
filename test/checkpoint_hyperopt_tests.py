import fesl
from fesl import printout
from data_repo_path import get_data_repo_path
import numpy as np
data_path = get_data_repo_path()+"Al256_reduced/"
checkpoint_name = "test_ho"


def original_setup(n_trials):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################
    test_parameters = fesl.Parameters()
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
    test_parameters.running.max_number_epochs = 10
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # Specify the number of trials, the hyperparameter optimizer should run
    # and the type of hyperparameter.
    test_parameters.hyperparameters.n_trials = n_trials
    test_parameters.hyperparameters.hyper_opt_method = "optuna"
    test_parameters.hyperparameters.checkpoints_each_trial = 5
    test_parameters.hyperparameters.checkpoint_name = checkpoint_name
    test_parameters.manual_seed = 1002

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    data_handler = fesl.DataHandler(test_parameters)

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

    test_hp_optimizer = fesl.HyperOptInterface(test_parameters, data_handler)

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

    # Perform hyperparameter optimization.
    printout("Starting Hyperparameter optimization.")
    # test_hp_optimizer.perform_study()
    # test_hp_optimizer.set_optimal_parameters()
    # printout("Hyperparameter optimization: DONE.")

    # ####################
    # # TRAINING
    # # Train with these new parameters.
    # ####################
    #
    # test_network = fesl.Network(test_parameters)
    # test_trainer = fesl.Trainer(test_parameters, test_network, data_handler)
    # printout("Network setup: DONE.")
    # test_trainer.train_network()
    # printout("Training: DONE.")
    return test_hp_optimizer


def resume_checkpoint():
    loaded_params, new_datahandler, new_hyperopt = \
        fesl.HyperOptOptuna.resume_checkpoint(checkpoint_name)
    return new_hyperopt


def run_hyperopt_checkpoint_test(accuracy=1e-14):
    # First run the entire test.
    hyperopt = original_setup(9)
    hyperopt.perform_study()
    original_final_test_value = hyperopt.study.best_trial.value

    # Now do the same, but cut at epoch 22 and see if it recovers the
    # correct result.
    hyperopt = original_setup(9)
    hyperopt.perform_study()
    hyperopt = resume_checkpoint()
    hyperopt.perform_study()
    new_final_test_value = hyperopt.study.best_trial.value

    if np.abs(original_final_test_value-new_final_test_value) > accuracy:
        print("test_general did not suceed.", original_final_test_value,
              new_final_test_value)
        return False
    return True


if __name__ == "__main__":
    test1 = run_hyperopt_checkpoint_test()
    printout("Checkpoint test - success?:", test1)
