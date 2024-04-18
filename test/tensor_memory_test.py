import os

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

# Define the accuracy used in the tests.
accuracy = 1e-5


class TestTensorMemory:
    """
    Tests/Tries out how memory referencing/copying between np and torch works.

    This test does not necessarily have to be run all the time, I thought of
    it more as something that comes in handy when debugging while something
    breaks after an update. MALA relies on the following assumptions to
    be true.
    """

    def test_tensor_memory(self):

        # Load an array as a numpy array
        loaded_array_raw = np.load(
            os.path.join(data_path, "Be_snapshot0.in.npy")
        )

        # Get dimensions of numpy array.
        dimension = np.shape(loaded_array_raw)
        datacount = dimension[0] * dimension[1] * dimension[2]

        # Reshape as datacount x featurelength
        loaded_array_raw = loaded_array_raw.astype(np.float32)
        loaded_array = loaded_array_raw.reshape([datacount, dimension[3]])

        # Check if reshaping allocated new memory.
        loaded_array_raw *= 10
        assert np.isclose(
            np.sum(loaded_array), np.sum(loaded_array_raw), accuracy
        )

        # simulate data splitting.
        index1 = int(80 / 100 * np.shape(loaded_array)[0])
        torch_tensor = torch.from_numpy(loaded_array[0:index1]).float()

        # Check if tensor and array are still the same.
        assert np.isclose(
            torch.sum(torch_tensor), np.sum(loaded_array[0:index1]), accuracy
        )

        # Simulate data operation.
        loaded_array *= 10

        # Check if tensor and array are still the same.
        test1 = torch.abs(torch.sum(torch_tensor - loaded_array[0:index1]))
        assert np.isclose(
            torch.sum(torch_tensor), np.sum(loaded_array[0:index1]), accuracy
        )

        # Simulate Tensor data handling in pytorch workflow.
        data_set = TensorDataset(torch_tensor, torch_tensor)
        data_loader = DataLoader(data_set, batch_size=index1)

        # Perform data operation again.
        loaded_array *= 10
        for x, y in data_loader:
            assert np.isclose(
                torch.sum(x), np.sum(loaded_array[0:index1]), accuracy
            )
