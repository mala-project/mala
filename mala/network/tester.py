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
        # Forward through network.
        return self.\
            _forward_entire_snapshot(snapshot_number,
                                     self.data.test_data_set,
                                     self.number_of_batches_per_snapshot,
                                     self.parameters.mini_batch_size)
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
            inputs = inputs.to(f"{self.parameters_full.device_type}:"
                               f"{self.parameters_full.device_id}")
            predicted_outputs[i * self.parameters.
                              mini_batch_size:(i + 1) * self.parameters.
                              mini_batch_size, :] = \
                self.data.output_data_scaler.\
                inverse_transform(self.network(inputs).
                                  to('cpu'), as_numpy=True)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        predicted_outputs = self.data.target_calculator.\
            restrict_data(predicted_outputs)
        return actual_outputs, predicted_outputs

    def __prepare_to_test(self):
        """Prepare the tester class to for test run."""
        # We will use the DataSet iterator to iterate over the test data.
        # But since we only want the data per snapshot,
        # we need to make sure the batch size is compatible with that.
        # TODO: Check if this is still needed.
        # if self.parameters_full.use_horovod:
        #     self.parameters.sampler["test_sampler"] = torch.utils.data. \
        #         distributed.DistributedSampler(self.data.test_data_set,
        #                                        num_replicas=hvd.size(),
        #                                        rank=hvd.rank())
        #     self.test_data_loader = DataLoader(self.data.test_data_set,
        #                                        batch_size=self.parameters.
        #                                        mini_batch_size * 1,
        #                                        sampler=self.parameters.
        #                                        sampler["test_sampler"],
        #                                        **self.parameters.kwargs,
        #                                        shuffle=False)

        optimal_batch_size = self.\
            _correct_batch_size_for_testing(self.data.grid_size,
                                            self.parameters.mini_batch_size)
        if optimal_batch_size != self.parameters.mini_batch_size:
            printout("Had to readjust batch size from",
                     self.parameters.mini_batch_size, "to",
                     optimal_batch_size)
            self.parameters.mini_batch_size = optimal_batch_size
        self.number_of_batches_per_snapshot = int(self.data.grid_size /
                                                  self.parameters.
                                                  mini_batch_size)
