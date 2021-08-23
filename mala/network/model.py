from abc import abstractmethod, ABC
import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    def __init__(self, params):
        # copy the models params from the input parameter object
        self.use_horovod = params.use_horovod
        self.params = params.model

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # initialize the parent class
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, inputs):
        pass

    # TODO: Do we actually need this?
    # @abstractmethod
    # def do_prediction(self, array):
    #     pass

    @abstractmethod
    def calculate_loss(self, output, target):
        pass

