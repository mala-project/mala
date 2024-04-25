import os

import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")
test_checkpoint_name = "test"

# Define the accuracy used in the tests.
accuracy = 1e-14


class TestTrainingCheckpoint:
    """Tests the checkpointing capabilities of the NN training."""

    def test_general(self):
        """Test that training checkpointing works in general."""
        # First run the entire test.
        trainer = self.__original_setup(test_checkpoint_name, 40)
        trainer.train_network()
        original_final_test_loss = trainer.final_test_loss

        # Now do the same, but cut at epoch 22 and see if it recovers the
        # correct result.
        trainer = self.__original_setup(test_checkpoint_name, 22)
        trainer.train_network()
        trainer = self.__resume_checkpoint(test_checkpoint_name, 40)
        trainer.train_network()
        new_final_test_loss = trainer.final_test_loss
        assert np.isclose(
            original_final_test_loss, new_final_test_loss, atol=accuracy
        )

    def test_learning_rate(self):
        """Test that the learning rate scheduler is correctly checkpointed."""
        # First run the entire test.
        trainer = self.__original_setup(
            test_checkpoint_name,
            40,
            learning_rate_scheduler="ReduceLROnPlateau",
            learning_rate=0.1,
        )
        trainer.train_network()
        original_learning_rate = trainer.optimizer.param_groups[0]["lr"]

        # Now do the same, but cut at epoch 22 and see if it recovers the
        # correct result.
        trainer = self.__original_setup(
            test_checkpoint_name,
            22,
            learning_rate_scheduler="ReduceLROnPlateau",
            learning_rate=0.1,
        )
        trainer.train_network()
        trainer = self.__resume_checkpoint(test_checkpoint_name, 40)
        trainer.train_network()
        new_learning_rate = trainer.optimizer.param_groups[0]["lr"]
        assert np.isclose(
            original_learning_rate, new_learning_rate, atol=accuracy
        )

    def test_early_stopping(self):
        """Test that the early stopping mechanism is correctly checkpointed."""
        # First run the entire test.
        trainer = self.__original_setup(
            test_checkpoint_name,
            40,
            early_stopping_epochs=30,
            learning_rate=0.1,
        )
        trainer.train_network()
        original_nr_epochs = trainer.last_epoch

        # Now do the same, but cut at epoch 22 and see if it recovers the
        # correct result.
        trainer = self.__original_setup(
            test_checkpoint_name,
            22,
            early_stopping_epochs=30,
            learning_rate=0.1,
        )
        trainer.train_network()
        trainer = self.__resume_checkpoint(test_checkpoint_name, 40)
        trainer.train_network()
        last_nr_epochs = trainer.last_epoch

        # integer comparison!
        assert original_nr_epochs == last_nr_epochs

    @staticmethod
    def __original_setup(
        checkpoint_name,
        maxepochs,
        learning_rate_scheduler=None,
        early_stopping_epochs=0,
        learning_rate=0.00001,
    ):
        """
        Sets up a NN training.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint to be used.

        maxepochs : int
            Maximum number of epochs.

        learning_rate_scheduler : string
            Type of learning rate scheduler.

        early_stopping_epochs : int
            Number of early stopping epochs.

        learning_rate : float
            Learning rate to be used.

        Returns
        -------
        trainer : mala.network.trainer.Trainer
            The trainer object created with the specified parameters.
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

        # Specify the used activation function.
        test_parameters.network.layer_activations = ["ReLU"]

        # Specify the training parameters.
        test_parameters.running.max_number_epochs = maxepochs
        test_parameters.running.mini_batch_size = 38
        test_parameters.running.learning_rate = learning_rate
        test_parameters.running.trainingtype = "Adam"
        test_parameters.running.learning_rate_scheduler = (
            learning_rate_scheduler
        )
        test_parameters.running.learning_rate_decay = 0.1
        test_parameters.running.learning_rate_patience = 30
        test_parameters.running.early_stopping_epochs = early_stopping_epochs
        test_parameters.running.early_stopping_threshold = -0.1
        test_parameters.running.checkpoints_each_epoch = 5
        test_parameters.running.checkpoint_name = checkpoint_name
        test_parameters.manual_seed = 1002
        test_parameters.running.use_shuffling_for_samplers = False

        ####################
        # DATA
        # Add and prepare snapshots for training.
        ####################

        data_handler = mala.DataHandler(test_parameters)

        # Add a snapshot we want to use in to the list.
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
        # NETWORK SETUP
        # Set up the network and trainer we want to use.
        # The layer sizes can be specified before reading data,
        # but it is safer this way.
        ####################

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )

        return test_trainer

    @staticmethod
    def __resume_checkpoint(checkpoint_name, actual_max_epochs):
        """
        Resumes a training from checkpoint.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint to be used.

        actual_max_epochs : int
            Maximum number of epochs to be used in total after recovering
            checkpoint.

        Returns
        -------
        trainer : mala.network.trainer.Trainer
            The trainer object created with the specified parameters.
        """

        loaded_params, loaded_network, new_datahandler, new_trainer = (
            mala.Trainer.load_run(checkpoint_name)
        )
        loaded_params.running.max_number_epochs = actual_max_epochs
        return new_trainer
