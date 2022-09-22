"""Runner class for running networks."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch

from mala.common.parameters import ParametersRunning
from mala import Parameters


class Runner:
    """
    Parent class for all classes that in some sense "run" the network.

    That can be training, benchmarking, inference, etc.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Runner object.

    network : mala.network.network.Network
        Network which is being run.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the run.
    """

    def __init__(self, params, network, data):
        self.parameters_full: Parameters = params
        self.parameters: ParametersRunning = params.running
        self.network = network
        self.data = data
        self.__prepare_to_run()

    def __prepare_to_run(self):
        """
        Prepare the Runner to run the Network.

        This includes e.g. horovod setup.
        """
        # See if we want to use horovod.
        if self.parameters_full.use_horovod:
            if self.parameters_full.use_gpu:
                # We cannot use "printout" here because this is supposed
                # to happen on every rank.
                if self.parameters_full.verbosity >= 2:
                    print("size=", hvd.size(), "global_rank=", hvd.rank(),
                          "local_rank=", hvd.local_rank(), "device=",
                          torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())

    def _forward_entire_snapshot(self, snapshot_number, data_set,
                                 number_of_batches_per_snapshot=0,
                                 batch_size=0):
        """
        Forward a snapshot through the network, get actual/predicted output.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the prediction is done.

        number_of_batches_per_snapshot : int
            Number of batches that lie within a snapshot.

        batch_size : int
            Batch size used for forward pass.

        Returns
        -------
        actual_outputs : numpy.ndarray
            Actual outputs for snapshot.

        predicted_outputs : numpy.ndarray
            Precicted outputs for snapshot.
        """
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = True
            actual_outputs = \
                (data_set
                 [snapshot_number * self.data.
                     grid_size:(snapshot_number + 1) * self.data.grid_size])[1]
        else:
            actual_outputs = \
                self.data.output_data_scaler.\
                inverse_transform(
                    (data_set[snapshot_number * self.data.grid_size:
                              (snapshot_number + 1) * self.data.grid_size])[1],
                    as_numpy=True)

        predicted_outputs = np.zeros((self.data.grid_size,
                                      self.data.get_output_dimension()))

        offset = snapshot_number * self.data.grid_size
        for i in range(0, number_of_batches_per_snapshot):
            inputs, outputs = \
                data_set[offset+(i * batch_size):offset+((i + 1) * batch_size)]
            inputs = inputs.to(self.parameters._configuration["device"])
            predicted_outputs[i * batch_size:(i + 1) * batch_size, :] = \
                self.data.output_data_scaler.\
                inverse_transform(self.network(inputs).
                                  to('cpu'), as_numpy=True)

        # Restricting the actual quantities to physical meaningful values,
        # i.e. restricting the (L)DOS to positive values.
        predicted_outputs = self.data.target_calculator.\
            restrict_data(predicted_outputs)

        # It could be that other operations will be happening with the data
        # set, so it's best to reset it.
        if self.data.parameters.use_lazy_loading:
            data_set.return_outputs_directly = False

        return actual_outputs, predicted_outputs

    @staticmethod
    def _correct_batch_size_for_testing(datasize, batchsize):
        """
        Get the correct batch size for testing.

        For testing snapshot the batch size needs to be such that
        data_per_snapshot / batch_size will result in an integer division
        without any residual value.
        """
        new_batch_size = batchsize
        if datasize % new_batch_size != 0:
            while datasize % new_batch_size != 0:
                new_batch_size += 1
        return new_batch_size
