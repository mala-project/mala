"""A special type of tensor data set for improved performance."""

import numpy as np
import torch


class FastTensorDataset(torch.utils.data.Dataset):
    """
    A special type of tensor data set for improved performance.

    This version of TensorDataset gathers data using a single call
    within  __getitem__. A bit more tricky to manage but is faster still.

    Parameters
    ----------
    batch_size : int
        Batch size to be used with this data set.

    tensors : object
        Torch tensors for this data set.

    Attributes
    ----------
    batch_size : int
        Batch size to be used with this data set.
    """

    def __init__(self, batch_size, *tensors):
        super(FastTensorDataset).__init__()
        self.batch_size = batch_size
        self._tensors = tensors
        total_samples = tensors[0].shape[0]
        self._indices = np.arange(total_samples)
        self._len = total_samples // self.batch_size

    def __getitem__(self, idx):
        """
        Get an item from the data set.

        This returns an entire batch

        Parameters
        ----------
        idx : int
            Batch Index.

        Returns
        -------
        batch : tuple
            The data tuple for this batch.
        """
        batch = self._indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        rv = tuple(t[batch, ...] for t in self._tensors)
        return rv

    def __len__(self):
        """Get the length of the data set."""
        return self._len

    def shuffle(self):
        """Shuffle the data set."""
        np.random.shuffle(self._indices)
