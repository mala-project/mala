"""Neural network for MALA."""
import torch
import torch.nn as nn
import torch.nn.functional as functional
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by parameters class
    pass


class Network(nn.Module):
    """Central network class for this framework, based on pytorch.nn.Module.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this neural network.
    """

    def __init__(self, params):
        # copy the network params from the input parameter object
        self.use_horovod = params.use_horovod
        self.params = params.network

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # initialize the parent class
        super(Network, self).__init__()

        # Mappings for parsing of the activation layers.
        self.activation_mappings = {
            "Sigmoid": nn.Sigmoid,
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU
        }

        # initialize the layers
        self.number_of_layers = len(self.params.layer_sizes) - 1
        self.layers = nn.ModuleList()

        if self.params.nn_type == "feed-forward":
            self.__initialize_as_feedforward()
        else:
            raise Exception("Unsupported network architecture.")

        # initialize the loss function
        if self.params.loss_function_type == "mse":
            self.loss_func = functional.mse_loss
        else:
            raise Exception("Unsupported loss function.")

        # Once everything is done, we can move the Network on the target
        # device.
        if params.use_gpu:
            self.to('cuda')

    def __initialize_as_feedforward(self):
        """Initialize this network as a feed-forward network."""
        # Check if multiple types of activations were selected or only one
        # was passed to be used in the entire network.#
        # If multiple layers have been passed, their size needs to be correct.

        use_only_one_activation_type = False
        if len(self.params.layer_activations) == 1:
            use_only_one_activation_type = True
        elif len(self.params.layer_activations) < self.number_of_layers:
            raise Exception("Not enough activation layers provided.")

        # Add the layers.
        # As this is a feedforward layer we always add linear layers, and then
        # an activation function
        for i in range(0, self.number_of_layers):
            self.layers.append((nn.Linear(self.params.layer_sizes[i],
                                          self.params.layer_sizes[i + 1])))
            try:
                if use_only_one_activation_type:
                    self.layers.append(self.activation_mappings[self.params.
                                       layer_activations[0]]())
                else:
                    self.layers.append(self.activation_mappings[self.params.
                                       layer_activations[i]]())
            except KeyError:
                raise Exception("Invalid activation type seleceted.")

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
        if self.params.nn_type == "feed-forward":
            for layer in self.layers:
                inputs = layer(inputs)
            return inputs
        else:
            raise Exception("Unsupported network architecture.")

    def do_prediction(self, array):
        """
        Predict the output values for an input array..

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

    # FIXME: This guarentees downwards compatibility, but it is ugly.
    #  Rather enforce the right package versions in the repo.
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
        # If we use horovod, only save the network on root.
        if self.use_horovod:
            if hvd.rank() != 0:
                return
        torch.save(self.state_dict(), path_to_file,
                   _use_new_zipfile_serialization=False)

    @classmethod
    def load_from_file(cls, params, path_to_file):
        """
        Load a network from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the network should be created.
            Has to be compatible to the network architecture. This is usually
            enforced by using the same Parameters object (and saving/loading
            it to)

        path_to_file : string
            Path to the file from which the network should be loaded.

        Returns
        -------
        loaded_network : Network
            The network that was loaded from the file.
        """
        loaded_network = Network(params)
        loaded_network.load_state_dict(torch.load(path_to_file))
        loaded_network.eval()
        return loaded_network
