import torch
from fesl.common.parameters import printout
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass


class Runner:
    """
    Parent class for all classes that in some sense "run" the network. That can be training, benchmarking,
    inference, e.g.
    """
    def __init__(self, p):
        self.parameters = p.running
        self.network = None
        self.batch_size=p.running.mini_batch_size
        self.use_gpu = p.use_gpu
        self.use_horovod=p.use_horovod
        self.use_compression = self.parameters.use_compression

    def prepare_to_run(self):
        # See if we want to use horovod.
        if self.use_horovod:
            if self.use_gpu:
                printout("size=", hvd.size(), "global_rank=", hvd.rank(), "local_rank=", hvd.local_rank(), "device=",
                      torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())
