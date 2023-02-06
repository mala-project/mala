import os
import importlib

import mala
import numpy as np
import pytest

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

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
        data_shuffler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                   "Be_snapshot0.out.npy", data_path)
        data_shuffler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                   "Be_snapshot1.out.npy", data_path)

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_shuffled*")

        test_parameters.data.shuffling_seed = 1234
        data_shuffler = mala.DataShuffler(test_parameters)

        # Add a snapshot we want to use in to the list.
        data_shuffler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                   "Be_snapshot0.out.npy", data_path)
        data_shuffler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                   "Be_snapshot1.out.npy", data_path)

        # After shuffling, these snapshots can be loaded as regular snapshots
        # for lazily loaded training-
        data_shuffler.shuffle_snapshots("./", save_name="Be_REshuffled*")

        old = np.load("Be_shuffled1.out.npy")
        new = np.load("Be_REshuffled1.out.npy")
        assert np.isclose(np.sum(np.abs(old-new)), 0.0, atol=accuracy)
