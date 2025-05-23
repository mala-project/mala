"""Neural network for MALA."""

from abc import abstractmethod
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

from mala.common.parameters import Parameters
from mala.common.parallelizer import printout, parallel_warn


class Network(nn.Module):
    """
    Central network class for this framework, based on pytorch.nn.Module.

    The correct type of neural network will automatically be instantiated
    by this class if possible. You can also instantiate the desired
    network directly by calling upon the subclass.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this neural network.

    Attributes
    ----------
    loss_func : function
        Loss function.

    mini_batch_size : int
        Size of mini batches propagated through network.

    number_of_layers : int
        Number of NN layers.

    params : mala.common.parametes.ParametersNetwork
        MALA neural network parameters.

    use_ddp : bool
        If True, the torch distributed data parallel formalism will be used.
    """

    def __new__(cls, params: Parameters = None):
        """
        Create a neural network instance.

        The correct type of neural network will automatically be instantiated
        by this class if possible. You can also instantiate the desired
        network directly by calling upon the subclass.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this neural network.
        """
        # Check if we're accessing through base class. If not, we need to
        # return the correct object directly.
        #
        # params=None if we load a serialized object, for instance. Then we
        # just want an empty object which gets populated during
        # deserialization.

        if params is None or cls != Network:
            model = super().__new__(cls)
        else:
            if params.network.nn_type == "feed-forward":
                model = super(Network, FeedForwardNet).__new__(FeedForwardNet)

            else:
                raise Exception("Unsupported network architecture.")

        return model

    def __init__(self, params: Parameters):
        # copy the network params from the input parameter object
        self.use_ddp = params.use_ddp
        self.params = params.network

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # initialize the parent class
        super(Network, self).__init__()

        # initialize the layers
        self.number_of_layers = len(self.params.layer_sizes) - 1

        # initialize the loss function
        if self.params.loss_function_type == "mse":
            self.loss_func = functional.mse_loss
        else:
            raise Exception("Unsupported loss function.")

    @abstractmethod
    def forward(self, inputs):
        """
        Abstract method. To be implemented by the derived class.

        Parameters
        ----------
        inputs : torch.Tensor
            Torch tensor to be propagated.
        """
        pass

    def do_prediction(self, array):
        """
        Predict the output values for an input array.

        Interface to do predictions. The data put in here is assumed to be a
        scaled torch.Tensor and in the right units. Be aware that this will
        pass the entire array through the network, which might be very
        demanding in terms of RAM.

        Parameters
        ----------
        array : torch.Tensor
            Input array for which the prediction is to be performed.

        Returns
        -------
        predicted_array : torch.Tensor
            Predicted outputs of array.

        """
        self.eval()
        with torch.no_grad():
            return self(array)

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
        return self.loss_func(output, target)

    def save_network(self, path_to_file):
        """
        Save the network.

        This function serves as an interfaces to pytorchs own saving
        functionalities AND possibly own saving needs.

        Parameters
        ----------
        path_to_file : string
            Path to the file in which the network should be saved.
        """
        # If we use ddp, only save the network on root.
        if self.use_ddp:
            if dist.get_rank() != 0:
                return
        torch.save(
            self.state_dict(),
            path_to_file,
            _use_new_zipfile_serialization=False,
        )

    @classmethod
    def load_from_file(cls, params, file):
        """
        Load a network from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the network should be created.
            Has to be compatible to the network architecture. This is usually
            enforced by using the same Parameters object (and saving/loading
            it to)

        file : string or ZipExtFile
            Path to the file from which the network should be loaded.

        Returns
        -------
        loaded_network : Network
            The network that was loaded from the file.
        """
        loaded_network = Network(params)
        loaded_network.load_state_dict(
            torch.load(file, map_location=params.device)
        )
        loaded_network.eval()
        return loaded_network


class FeedForwardNet(Network):
    """Initialize this network as a feed-forward network."""

    def __init__(self, params):
        super(FeedForwardNet, self).__init__(params)

        self.layers = nn.ModuleList()

        # If we have only one entry in the activation list,
        # we use it for the entire list.
        # We should NOT modify the list itself. This would break the
        # hyperparameter algorithms.
        use_only_one_activation_type = False
        if isinstance(self.params.layer_activations, list):
            if len(self.params.layer_activations) > self.number_of_layers:

                number_of_ignored_layers = (
                    len(self.params.layer_activations) - self.number_of_layers
                )
                number_of_ignored_layers += (
                    1
                    if self.params.layer_activations_include_output_layer
                    is False
                    else 0
                )
                printout(
                    "Too many activation layers provided. The last",
                    str(number_of_ignored_layers),
                    "activation function(s) will be ignored.",
                    min_verbosity=1,
                )

        # Add the layers.
        # As this is a feedforward NN we always add linear layers, and then
        # an activation function
        for i in range(0, self.number_of_layers):
            self.layers.append(
                (
                    nn.Linear(
                        self.params.layer_sizes[i],
                        self.params.layer_sizes[i + 1],
                    )
                )
            )
            if (
                i < self.number_of_layers - 1
            ) or self.params.layer_activations_include_output_layer:
                try:
                    if isinstance(self.params.layer_activations, list):
                        self._append_activation_function(
                            self.params.layer_activations[i]
                        )
                    else:
                        self._append_activation_function(
                            self.params.layer_activations
                        )

                except KeyError:
                    raise Exception("Invalid activation type seleceted.")
                except IndexError:
                    # No activation functions left to append at the end.
                    pass

        # Once everything is done, we can move the Network on the target
        # device.
        self.to(self.params._configuration["device"])

    def forward(self, inputs):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input array for which the forward pass is to be performed.

        Returns
        -------
        predicted_array : torch.Tensor
            Predicted outputs of array.
        """
        # Forward propagate data.
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def _append_activation_function(self, activation_function):
        """
        Append an activation function to the network.

        Parameters
        ----------
        activation_function : str or nn.Module or class
            Activation function to be appended.
        """
        if activation_function is None:
            pass
        elif isinstance(activation_function, str):
            try:
                self.layers.append(getattr(torch.nn, activation_function)())
            except AttributeError:
                raise Exception(
                    "Torch does not contain the specified "
                    "activation function: " + activation_function
                )
        elif isinstance(activation_function, nn.Module):
            self.layers.append(activation_function)
        elif issubclass(activation_function, nn.Module):
            self.layers.append(activation_function())
