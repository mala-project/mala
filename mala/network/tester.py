"""Tester class for testing a network."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import torch

from mala.common.parameters import printout
from mala.network.runner import Runner


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
        # Adjust batch size and such.
        self.__prepare_to_test(snapshot_number)

        # Forward through network.
        return self.\
            _forward_entire_snapshot(snapshot_number,
                                     self.data.test_data_set,
                                     "te",
                                     self.number_of_batches_per_snapshot,
                                     self.parameters.mini_batch_size)

    def __prepare_to_test(self, snapshot_number):
        """Prepare the tester class to for test run."""
        # We will use the DataSet iterator to iterate over the test data.
        # But since we only want the data per snapshot,
        # we need to make sure the batch size is compatible with that.
        test_snapshot = 0
        grid_size = None
        for snapshot in self.data.parameters.snapshot_directories_list:
            if snapshot.snapshot_function == "te":
                if snapshot_number == test_snapshot:
                    grid_size = snapshot.grid_size
                    break
                test_snapshot += 1

        optimal_batch_size = self.\
            _correct_batch_size_for_testing(grid_size,
                                            self.parameters.mini_batch_size)
        if optimal_batch_size != self.parameters.mini_batch_size:
            printout("Had to readjust batch size from",
                     self.parameters.mini_batch_size, "to",
                     optimal_batch_size, min_verbosity=0)
            self.parameters.mini_batch_size = optimal_batch_size
        self.number_of_batches_per_snapshot = int(grid_size /
                                                  self.parameters.
                                                  mini_batch_size)
