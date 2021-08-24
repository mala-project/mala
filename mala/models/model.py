from abc import abstractmethod, ABC
import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    """
    Base class of all models for MALA, based on pytorch.nn.Module.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this model.
    """

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
        """
        Perform a forward pass via the model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input array for which the forward pass is to be performed.

        Returns
        -------
        predicted_array : torch.Tensor
            Predicted outputs of array.
        """
        pass

    # TODO: Do we actually need this?
    # @abstractmethod
    # def do_prediction(self, array):
    #     pass

    @abstractmethod
    def calculate_loss(self, output, target):
        """
        Calculate the loss for a predicted output and target.

        Parameters
        ----------
        output : torch.Tensor
            Predicted output.

        target : torch.Tensor.
            Actual output.

        Returns
        -------
        loss_val : float
            Loss value for output and target.

        """
        pass

    @abstractmethod
    def save_model(self, path_to_file):
        """
        Save the model to file

        This function serves as an interfaces to pytorchs own saving
        functionalities AND possibly own saving needs.

        Parameters
        ----------
        path_to_file : string
            Path to the file in which the models should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_file(cls, params, path_to_file):
        """
        Load the model from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the models should be created.
            Has to be compatible to the models architecture. This is usually
            enforced by using the same Parameters object (and saving/loading
            it to)

        path_to_file : string
            Path to the file from which the models should be loaded.

        Returns
        -------
        loaded_network : Network
            The models that was loaded from the file.
        """
        pass
