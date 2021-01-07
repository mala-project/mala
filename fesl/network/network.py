import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    """Central network class for this framework. Based on pytorch.nn.Module."""

    def __init__(self, p):
        # copy the network params from the input parameter object
        self.params = p.network
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

        # TODO: Deconvolute this into their own classes.
        if self.params.nn_type == "feed-forward":
            self.initialize_as_feedforward()
        else:
            raise Exception("Unsupported network architecture.")

        # initialize the loss function
        if self.params.loss_function_type == "mse":
            self.loss_func = F.mse_loss
        else:
            raise Exception("Unsupported loss function.")

    def initialize_as_feedforward(self):
        # Check if multiple types of activations were selected or only one was passed to be used in the entire network.#
        # If multiple layers have been passed, their size needs to be correct.

        use_only_one_activation_type = False
        if len(self.params.layer_activations) == 1:
            use_only_one_activation_type = True
        elif len(self.params.layer_activations) < self.number_of_layers:
            raise Exception("Not enough activation layers provided.")

        # Add the layers.
        # As this is a feedforward layer we always add linear layers, and then an activation function
        for i in range(0, self.number_of_layers):
            self.layers.append((nn.Linear(self.params.layer_sizes[i], self.params.layer_sizes[i + 1])))
            try:
                if use_only_one_activation_type:
                    self.layers.append(self.activation_mappings[self.params.layer_activations[0]]())
                else:
                    self.layers.append(self.activation_mappings[self.params.layer_activations[i]]())
            except:
                raise Exception("Invalid activation type seleceted.")

    def forward(self, inputs):

        # Forward propagate data.
        if self.params.nn_type == "feed-forward":
            for layer in self.layers:
                inputs = layer(inputs)
            return inputs
        else:
            raise Exception("Unsupported network architecture.")

    def do_prediction(self, array):
        """
        Interface to do predictions. The data put in here is assumed to be scaled and in the right units.
        So far it is mostly an interface to pytorch, but maybe at a later date it will be more.
        """
        self.eval()
        return self(array)

    def calculate_loss(self, output, target):
        return self.loss_func(output, target)

    def save_network(self, path_to_file):
        """
        Saves the network. This function serves as an interfaces to pytorchs own saving functionalities
        AND possibly own saving needs.
        """
        torch.save(self.state_dict(), path_to_file)

    @classmethod
    def load_from_file(cls, params, path_to_file):
        """
        Loads a network from a file.
        """
        loaded_network = Network(params)
        loaded_network.load_state_dict(torch.load(path_to_file))
        loaded_network.eval()
        return loaded_network

    # FIXME: Move to a different file, as this is only needed for
    # classification problems and the LDOS prediction is no classification
    # problem. Also it is slow.

    @staticmethod
    def classification_accuracy(prediction, target):
        prediction_arguments = torch.argmax(prediction, dim=1)
        target_arguments = torch.argmax(target, dim=1)
        return ((prediction_arguments == target_arguments).float()).mean()
