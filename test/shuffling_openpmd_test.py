import os
import importlib

import mala
import numpy as np
import pytest
import openpmd_api as io

# Accuracy for the shuffling test.
accuracy = np.finfo(float).eps

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

class TestDataShuffling:
    """Test that the MALA data shuffling works."""

    def __convert_to_openpmd():

        from mala.datahandling.data_repo import data_repo_path
        data_path = os.path.join(data_repo_path, "Be2")
        # Sneakily use the data shuffler for conversion
        # Temporary until we have openPMD sample data
        params = mala.Parameters()
        params.data.shuffling_seed = 1234
        params.openpmd_configuration = {
            'adios2': {
                'engine': {
                    'parameters': {
                        'StatsLevel': 1
                    }
                }
            }
        }
        data_shuffler = mala.DataShuffler(params)
        data_shuffler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                   "Be_snapshot0.out.npy", data_path)
        data_shuffler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                   "Be_snapshot1.out.npy", data_path)
        data_shuffler.shuffle_snapshots(complete_save_path="./shuffled",
                                        save_name="Be_snapshot*.h5")

    def test_convert(self):
        TestDataShuffling.__convert_to_openpmd()

    def test_seed(self):
        """Test that the shuffling is handled correctly internally."""
        test_parameters = mala.Parameters()
        test_parameters.openpmd_configuration = {
            'adios2': {
                'engine': {
                    'parameters': {
                        'StatsLevel': 1
                    }
                }
            }
        }
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                   "Be_snapshot0.out.h5", data_path,
                                   snapshot_type="openpmd")
        data_shuffler.add_snapshot("Be_snapshot1.in.h5", data_path,
                                   "Be_snapshot1.out.h5", data_path,
                                   snapshot_type="openpmd")

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*.h5")

        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                   "Be_snapshot0.out.h5", data_path,
                                   snapshot_type="openpmd")
        data_shuffler.add_snapshot("Be_snapshot1.in.h5", data_path,
                                   "Be_snapshot1.out.h5", data_path,
                                   snapshot_type="openpmd")

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_REshuffled*.h5")

        old = io.Series("Be_shuffled1.out.h5", io.Access.read_only)
        new = io.Series("Be_REshuffled1.out.h5", io.Access.read_only)
        for it_index, old_it in old.iterations.items():
            new_it = new.iterations[it_index]
            for mesh_name, old_mesh in old_it.meshes.items():
                new_mesh = new_it.meshes[mesh_name]
                for component_name, old_component in old_mesh.items():
                    new_component = new_mesh[component_name]
                    old_array = old_component[()]
                    new_array = new_component[()]
                    old.flush()
                    new.flush()
                    assert np.isclose(np.sum(
                        np.abs(old_array-new_array)), 0.0, atol=accuracy)

    def test_training(self):
        test_parameters = mala.Parameters()
        test_parameters.openpmd_configuration = {
            'adios2': {
                'engine': {
                    'parameters': {
                        'StatsLevel': 1
                    }
                }
            }
        }
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 50
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True

        # Train without shuffling.
        data_handler = mala.DataHandler(test_parameters)
        from mala.datahandling.data_repo import data_repo_path
        original_data_path = os.path.join(data_repo_path, "Be2")
        data_handler.add_snapshot("Be_snapshot0.in.h5", original_data_path,
                                  "Be_snapshot0.out.h5", original_data_path, "tr", snapshot_type='openpmd')
        data_handler.add_snapshot("Be_snapshot1.in.h5", original_data_path,
                                  "Be_snapshot1.out.h5", original_data_path, "va", snapshot_type='openpmd')
        data_handler.prepare_data()

        test_parameters.network.layer_sizes = [data_handler.input_dimension,
                                               100,
                                               data_handler.output_dimension]
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()
        old_loss = test_trainer.final_validation_loss

        # Shuffle.
        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                   "Be_snapshot0.out.h5", data_path,
                                   snapshot_type="openpmd")
        data_shuffler.add_snapshot("Be_snapshot1.in.h5", data_path,
                                   "Be_snapshot1.out.h5", data_path,
                                   snapshot_type="openpmd")

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*.h5")
        test_parameters.descriptors.descriptors_contain_xyz = True

        # Train with shuffling.
        data_handler = mala.DataHandler(test_parameters)
        # Add a snapshot we want to use in to the list.
        data_handler.add_snapshot("Be_shuffled0.in.h5", ".",
                                  "Be_shuffled0.out.h5", ".", "tr",
                                   snapshot_type="openpmd")
        data_handler.add_snapshot("Be_shuffled1.in.h5", ".",
                                  "Be_shuffled1.out.h5", ".", "va",
                                   snapshot_type="openpmd")
        data_handler.prepare_data()

        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()
        new_loss = test_trainer.final_validation_loss
        assert old_loss > new_loss

TestDataShuffling().test_training()