import os

import mala
import numpy as np

from mala.datahandling.data_repo import data_path

# Accuracy for the shuffling test.
accuracy = np.finfo(float).eps


class TestDataShuffling:
    """Test that the MALA data shuffling works."""

    def test_seed(self):
        """Test that the shuffling is handled correctly internally."""
        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*")

        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_REshuffled*")

        old = np.load("Be_shuffled1.out.npy")
        new = np.load("Be_REshuffled1.out.npy")
        assert np.isclose(np.sum(np.abs(old - new)), 0.0, atol=accuracy)

    def test_seed_openpmd(self):
        """
        Test that the shuffling is handled correctly internally.

        This function tests the shuffling for OpenPMD and confirms that
        shuffling both from numpy and openpmd into openpmd always gives the
        same results. The first shuffling shuffles from openpmd to openpmd
        format, the second from numpy to openpmd.
        """
        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.h5",
            data_path,
            "Be_snapshot0.out.h5",
            data_path,
            snapshot_type="openpmd",
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.h5",
            data_path,
            "Be_snapshot1.out.h5",
            data_path,
            snapshot_type="openpmd",
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*.h5")

        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.npy",
            data_path,
            "Be_snapshot0.out.npy",
            data_path,
            snapshot_type="numpy",
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.npy",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
            snapshot_type="numpy",
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_REshuffled*.h5")

        old = data_shuffler.target_calculator.read_from_openpmd_file(
            "Be_shuffled1.out.h5"
        )
        new = data_shuffler.target_calculator.read_from_openpmd_file(
            "Be_REshuffled1.out.h5"
        )
        assert np.isclose(np.sum(np.abs(old - new)), 0.0, atol=accuracy)

    def test_training(self):
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 50
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True

        # Train without shuffling.
        data_handler = mala.DataHandler(test_parameters)
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
        old_loss = test_trainer.final_validation_loss

        # Shuffle.
        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 50
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*")
        test_parameters.descriptors.descriptors_contain_xyz = True

        # Train with shuffling.
        data_handler = mala.DataHandler(test_parameters)
        # Add a snapshot we want to use in to the list.
        data_handler.add_snapshot(
            "Be_shuffled0.in.npy", ".", "Be_shuffled0.out.npy", ".", "tr"
        )
        data_handler.add_snapshot(
            "Be_shuffled1.in.npy", ".", "Be_shuffled1.out.npy", ".", "va"
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
        new_loss = test_trainer.final_validation_loss
        assert old_loss > new_loss

    def test_training_openpmd(self):
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 50
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True

        # Train without shuffling.
        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot(
            "Be_snapshot0.in.h5",
            data_path,
            "Be_snapshot0.out.h5",
            data_path,
            "tr",
            snapshot_type="openpmd",
        )
        data_handler.add_snapshot(
            "Be_snapshot1.in.h5",
            data_path,
            "Be_snapshot1.out.h5",
            data_path,
            "va",
            snapshot_type="openpmd",
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
        old_loss = test_trainer.final_validation_loss

        # Shuffle.
        test_parameters = mala.Parameters()
        test_parameters.data.shuffling_seed = 1234
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 50
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True

        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot(
            "Be_snapshot0.in.h5",
            data_path,
            "Be_snapshot0.out.h5",
            data_path,
            snapshot_type="openpmd",
        )
        data_shuffler.add_snapshot(
            "Be_snapshot1.in.h5",
            data_path,
            "Be_snapshot1.out.h5",
            data_path,
            snapshot_type="openpmd",
        )

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*.h5")
        test_parameters.descriptors.descriptors_contain_xyz = True

        # Train with shuffling.
        data_handler = mala.DataHandler(test_parameters)
        # Add a snapshot we want to use in to the list.
        data_handler.add_snapshot(
            "Be_shuffled0.in.h5",
            ".",
            "Be_shuffled0.out.h5",
            ".",
            "tr",
            snapshot_type="openpmd",
        )
        data_handler.add_snapshot(
            "Be_shuffled1.in.h5",
            ".",
            "Be_shuffled1.out.h5",
            ".",
            "va",
            snapshot_type="openpmd",
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
        new_loss = test_trainer.final_validation_loss
        assert old_loss > new_loss

    def worker_arbitrary_number_snapshots(
        self, ext_in, ext_out, snapshot_type
    ):
        parameters = mala.Parameters()

        # This ensures reproducibility of the created data sets.
        parameters.data.shuffling_seed = 1234

        data_shuffler = mala.DataShuffler(parameters)

        for i in range(5):
            data_shuffler.add_snapshot(
                f"Be_snapshot0.in.{ext_in}",
                data_path,
                f"Be_snapshot0.out.{ext_in}",
                data_path,
                snapshot_type=snapshot_type,
            )
        data_shuffler.shuffle_snapshots(
            complete_save_path=".",
            save_name=f"Be_shuffled*{ext_out}",
            number_of_shuffled_snapshots=5,
        )

    def test_arbitrary_number_snapshots(self):
        self.worker_arbitrary_number_snapshots("npy", "", "numpy")
        for i in range(5):
            bispectrum = np.load("Be_shuffled" + str(i) + ".in.npy")
            ldos = np.load("Be_shuffled" + str(i) + ".out.npy")
            assert not np.any(np.where(np.all(ldos == 0, axis=-1).squeeze()))
            assert not np.any(
                np.where(np.all(bispectrum == 0, axis=-1).squeeze())
            )
        self.worker_arbitrary_number_snapshots("h5", ".h5", "openpmd")
        import openpmd_api as opmd

        bispectrum_series = opmd.Series(
            "Be_shuffled%T.in.h5", opmd.Access.read_only
        )
        ldos_series = opmd.Series(
            "Be_shuffled%T.out.h5", opmd.Access.read_only
        )
        for i in range(5):
            for name, series in [("Bispectrum", bispectrum_series), ("LDOS", ldos_series)]:
                loaded_array = [
                    component.load_chunk().squeeze()
                    for _, component in series.iterations[i]
                    .meshes[name]
                    .items()
                ]
                series.flush()
                loaded_array = np.array(loaded_array)
                assert not np.any(
                    np.where(np.all(loaded_array == 0, axis=0).squeeze())
                )
