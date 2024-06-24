import os

import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_path

checkpoint_name = "test_ho"

# Define the accuracy used in the tests.
accuracy = 1e-14


class TestHyperoptCheckpointing:
    """Tests the checkpointing capabilities of the hyperparam optimization."""

    def test_hyperopt_checkpoint(self):
        # First run the entire test.
        hyperopt = self.__original_setup(9)
        hyperopt.perform_study()
        original_final_test_value = hyperopt.study.best_trial.value

        # Now do the same, but cut at epoch 22 and see if it recovers the
        # correct result.
        hyperopt = self.__original_setup(9)
        hyperopt.perform_study()
        hyperopt = self.__resume_checkpoint()
        hyperopt.perform_study()
        new_final_test_value = hyperopt.study.best_trial.value

        assert np.isclose(
            original_final_test_value, new_final_test_value, atol=accuracy
        )

    @staticmethod
    def __original_setup(n_trials):
        """
        Perform original setup for HyperOpt.

        Parameters
        ----------
        n_trials : int
            Number of trials to perform.

        Returns
        -------
        hyperopt: mala.network.hyper_opt.HyperOpt:
            The hyperopt object.

        """
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
        test_parameters.running.optimizer = "Adam"

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
        data_handler = mala.DataHandler(test_parameters)

        # Add all the snapshots we want to use in to the list.
        data_handler.add_snapshot(
            "Be_snapshot0.in.npy",
            data_path,
            "Be_snapshot0.out.npy",
            data_path,
            "tr",
        )
        data_handler.add_snapshot(
            "Be_snapshot1.in.npy",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
            "va",
        )
        data_handler.add_snapshot(
            "Be_snapshot2.in.npy",
            data_path,
            "Be_snapshot2.out.npy",
            data_path,
            "te",
        )
        data_handler.prepare_data()
        printout("Read data: DONE.", min_verbosity=0)

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
        test_hp_optimizer.add_hyperparameter(
            "float", "learning_rate", 0.0000001, 0.01
        )

        # Number of neurons per layer will be optimized.
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_00", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_01", 10, 100
        )

        # Choices for activation function at each layer will be optimized.
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )

        # Perform hyperparameter optimization.
        printout("Starting Hyperparameter optimization.", min_verbosity=0)

        return test_hp_optimizer

    @staticmethod
    def __resume_checkpoint():
        """
        Resume a HyperOpt from a checkpoint.

        Returns
        -------
        hyperopt: mala.network.hyper_opt.HyperOpt:
            The hyperopt object.

        """
        loaded_params, new_datahandler, new_hyperopt = (
            mala.HyperOptOptuna.resume_checkpoint(checkpoint_name)
        )
        return new_hyperopt
