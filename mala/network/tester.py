"""Tester class for testing a network."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from mala.common.parameters import printout
from .runner import Runner
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass


class Tester(Runner):
    """
    A class for testing a neural network.

    It enables easy inference throughout a test set.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Tester object.

    network : mala.network.network.Network
        Network which is being tested.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the test data.
    """

    def __init__(self, params, network, data):
        # copy the parameters into the class.
        super(Tester, self).__init__(params, network, data)
        self.test_data_loader = None
        self.number_of_batches_per_snapshot = 0
        self.__prepare_to_test()

    def test_snapshot(self, snapshot_number):
        """
        Get actual and predicted output for a snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the prediction is done.

        Returns
        -------
        actual_outputs : torch.Tensor
            Actual outputs for snapshot.

        predicted_outputs : torch.Tensor
            Precicted outputs for snapshot.
        """
        self.data.prepare_for_testing()
        if self.data.parameters.use_lazy_loading:
            actual_outputs = \
                (self.data.test_data_set
                 [snapshot_number * self.data.
                     grid_size:(snapshot_number + 1) * self.data.grid_size])[1]
        else:
            actual_outputs = \
                self.data.output_data_scaler.\
                inverse_transform(
                    (self.data.test_data_set[snapshot_number *
                                             self.data.grid_size:
                                             (snapshot_number + 1) *
                                             self.data.grid_size])[1],
                    as_numpy=True)

        predicted_outputs = np.zeros((self.data.grid_size,
                                      self.data.get_output_dimension()))

        offset = snapshot_number * self.data.grid_size
        for i in range(0, self.number_of_batches_per_snapshot):
            inputs, outputs = \
                self.data.test_data_set[offset+(i * self.parameters.
                                        mini_batch_size):
                                        offset+((i + 1) * self.parameters.
                                        mini_batch_size)]
            if self.parameters_full.use_gpu:
                inputs = inputs.to('cuda')
            predicted_outputs[i * self.parameters.
                              mini_batch_size:(i + 1) * self.parameters.
                              mini_batch_size, :] = \
                self.data.output_data_scaler.\
                inverse_transform(self.network(inputs).
                                  to('cpu'), as_numpy=True)

        return actual_outputs, predicted_outputs

    def __prepare_to_test(self):
        """Prepare the tester class to for test run."""
        if self.parameters_full.use_horovod:
            self.parameters.sampler["test_sampler"] = torch.utils.data.\
                distributed.DistributedSampler(self.data.test_data_set,
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank())
        # We will use the DataLoader iterator to iterate over the test data.
        # But since we only want the data per snapshot,
        # we need to make sure the batch size is compatible with that.
        self.__check_and_adjust_batch_size(self.data.grid_size)
        self.test_data_loader = DataLoader(self.data.test_data_set,
                                           batch_size=self.parameters.
                                           mini_batch_size * 1,
                                           sampler=self.parameters.
                                           sampler["test_sampler"],
                                           **self.parameters.kwargs,
                                           shuffle=False)
        self.number_of_batches_per_snapshot = int(self.data.grid_size /
                                                  self.parameters.
                                                  mini_batch_size)

    def __check_and_adjust_batch_size(self, datasize):
        """
        Check batch size and adjust it if necessary.

        For testing the batch size needs to be such that data_per_snapshot /
        batch_size will result in an integer division without any residual
        value.
        """
        if datasize % self.parameters.mini_batch_size != 0:
            old_batch_size = self.parameters.mini_batch_size
            while datasize % self.parameters.mini_batch_size != 0:
                self.parameters.mini_batch_size += 1
            printout("Had to readjust batch size from", old_batch_size, "to",
                     self.parameters.mini_batch_size)
