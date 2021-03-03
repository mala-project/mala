"""Runner class for running networks."""
import torch
from fesl.common.parameters import printout
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

    def __init__(self, params):
        """
        Create a Runner object to run a Network.

        Parameters
        ----------
        params : fesl.common.parametes.Parameters
            Parameters used to create this Runner object.
        """
        self.parameters_full = params
        self.parameters = params.running
        self.network = None
        self.data = None

    def prepare_to_run(self):
        """
        Prepare the Runner to run the Network.

        This includes e.g. horovod setup.
        """
        # See if we want to use horovod.
        if self.parameters_full.use_horovod:
            if self.parameters_full.use_gpu:
                printout("size=", hvd.size(), "global_rank=", hvd.rank(),
                         "local_rank=", hvd.local_rank(), "device=",
                         torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())

    def save_network_objects(self, save_path=None,
                             network_name="tmp_network.pth",
                             iscaler_name="tmp_iscaler.pkl",
                             oscaler_name="tmp_oscaler.pkl",
                             params_name="tmp_params.pkl"):
        """
        Saves all objects (except for self) needed for running a network.

        Useful as a convenience function to store results at the end of
        a training run.

        Parameters
        ----------
        save_path : string
            Path to which the objects are saved. If None, the
            names will be interpreted as absolute paths.

        network_name : string
            Name of the file the network should be saved to.

        iscaler_name : string
            Name of the file the input scaler should be saved to.

        oscaler_name : string
            Name of the file the output scaler should be saved to.

        params_name : string
            Name of the file the parameters should be saved to.
        """
        if save_path is None:
            self.network.save_network(network_name)
            self.data.input_data_scaler.save(iscaler_name)
            self.data.output_data_scaler.save(oscaler_name)
            self.parameters_full.save(params_name)
        else:
            self.network.save_network(save_path + network_name)
            self.data.input_data_scaler.save(save_path + iscaler_name)
            self.data.output_data_scaler.save(save_path + oscaler_name)
            self.parameters_full.save(save_path + params_name)
