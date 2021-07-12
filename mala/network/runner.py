"""Runner class for running networks."""
import torch
from mala.common.parameters import printout
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass


class Runner:
    """
    Parent class for all classes that in some sense "run" the network.

    That can be training, benchmarking, inference, etc.
    """

    def __init__(self, params, network, data):
        """
        Create a Runner object to run a Network.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this Runner object.

        network : mala.network.network.Network
            Network which is being run.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the data for the run.
        """
        self.parameters_full = params
        self.parameters = params.running
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
                print("size=", hvd.size(), "global_rank=", hvd.rank(),
                         "local_rank=", hvd.local_rank(), "device=",
                         torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())
