import os

import mala
import numpy as np

from mala.datahandling.data_repo import data_path

# Control the accuracies for the postprocessing routines.
accuracy_predictions = 1.0


class TestOnTheFly:
    """
    Test the on-the-fly descriptor calculation capabilities.

    This relates to training, testing and shuffling descriptor data that has
    been computed on the fly.
    """

    @staticmethod
    def __setup_training(lazy_loading, checkpoint_name=None):
        test_parameters = mala.Parameters()
        test_parameters.use_lammps = False
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.data.use_lazy_loading = lazy_loading
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 5
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        if checkpoint_name:
            test_parameters.running.checkpoints_each_epoch = 5
            test_parameters.running.checkpoint_name = checkpoint_name

        test_parameters.descriptors.descriptor_type = "Bispectrum"
        # These values make no phyiscal sense, but are chosen to make the test
        # fast enough to be executed without LAMMPS.
        test_parameters.descriptors.bispectrum_twojmax = 2
        test_parameters.descriptors.bispectrum_cutoff = 1.5

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot(
            "Be_snapshot0.info.json",
            data_path,
            "Be_snapshot0.out.npy",
            data_path,
            "tr",
        )
        data_handler.add_snapshot(
            "Be_snapshot1.info.json",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
            "va",
        )
        data_handler.prepare_data()

        # Train a network.
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
        loaded_params, loaded_network, new_datahandler, new_trainer = (
            mala.Trainer.load_run(checkpoint_name)
        )
        loaded_params.running.max_number_epochs = actual_max_epochs
        return new_trainer

    def test_training(self):
        test_trainer = self.__setup_training(False)
        test_trainer.train_network()
        assert test_trainer.final_validation_loss < np.inf

    def test_training_lazy_loading(self):
        test_trainer = self.__setup_training(True)
        test_trainer.train_network()
        assert test_trainer.final_validation_loss < np.inf

    def test_training_lazy_loading_interrupted(self):
        test_trainer = self.__setup_training(True, "test_checkpoint_onthefly")
        test_trainer.train_network()
        test_trainer.data.delete_temporary_inputs()
        test_trainer = self.__resume_checkpoint("test_checkpoint_onthefly", 10)
        test_trainer.train_network()
        assert test_trainer.final_validation_loss < np.inf

    def test_training_lazy_loading_uninterrupted(self):
        test_trainer = self.__setup_training(True, "test_checkpoint_onthefly")
        test_trainer.train_network()
        test_trainer = self.__resume_checkpoint("test_checkpoint_onthefly", 10)
        test_trainer.train_network()
        assert test_trainer.final_validation_loss < np.inf

    def test_shuffling(self):
        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 5
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True
        data_shuffler = mala.DataShuffler(test_parameters)
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        # These values make no phyiscal sense, but are chosen to make the test
        # fast enough to be executed without LAMMPS.
        test_parameters.descriptors.bispectrum_twojmax = 2
        test_parameters.descriptors.bispectrum_cutoff = 1.5

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.info.json",
            data_path,
            "Be_snapshot0.out.npy",
            data_path,
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.info.json",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots(
            "./", save_name="Be_shuffled*", shuffle_to_temporary=True
        )
        test_parameters.descriptors.descriptors_contain_xyz = True

        # Train with shuffling.
        data_handler = mala.DataHandler(test_parameters)
        # Add a snapshot we want to use in to the list.
        data_handler.add_snapshot(
            data_shuffler.temporary_shuffled_snapshots[0].input_npy_file,
            data_shuffler.temporary_shuffled_snapshots[0].input_npy_directory,
            data_shuffler.temporary_shuffled_snapshots[0].output_npy_file,
            data_shuffler.temporary_shuffled_snapshots[0].output_npy_directory,
            "tr",
        )
        data_handler.add_snapshot(
            data_shuffler.temporary_shuffled_snapshots[1].input_npy_file,
            data_shuffler.temporary_shuffled_snapshots[1].input_npy_directory,
            data_shuffler.temporary_shuffled_snapshots[1].output_npy_file,
            data_shuffler.temporary_shuffled_snapshots[1].output_npy_directory,
            "va",
        )
        data_handler.prepare_data()

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )
        test_trainer.train_network()
        assert test_trainer.final_validation_loss < np.inf
