'''
network.py contains the neural network class used in this framework.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class network(nn.Module):
    """Central network class for this framework. Based on pytorch.nn.Module."""
    def __init__(self, p):
        # copy the network params from the input parameter object
        self.params = p.network
        # initialize the parent class
        super(network, self).__init__()

        # initialize the layers
        self.number_of_layers = len(self.params.layer_sizes)
        self.layers = nn.ModuleList()
        if (self.params.nn_type == "feed-forward"):
            self.initialize_as_feedforward()
        else:
            raise Exception("Unsupported network architecture.")

        # initialize the loss function
        if (self.params.loss_function_type == "mse"):
            self.loss_func = F.mse_loss
        else:
            raise Exception("Unsupported loss function.")


    def initialize_as_feedforward(self):
        for i in range(0,self.number_of_layers-1):
            self.layers.append((nn.Linear(self.params.layer_sizes[i], self.params.layer_sizes[i+1])))
            self.layers.append(nn.Sigmoid())


    def forward(self, input):
        if (self.params.nn_type == "feed-forward"):
            for layer in self.layers:
                input = layer(input)
            return input
        else:
            raise Exception("Unsupported network architecture.")

    def calculate_loss(self, output, target):
        return self.loss_func(output, target)

    # FIXME: Move to a different file, as this is only needed for
    # classification problems and the LDOS prediction is no classification
    # problem. Also it is slow.

    def classification_accuracy(self, prediction, target):
            prediction_arguments = torch.argmax(prediction, dim=1)
            target_arguments = torch.argmax(target, dim=1)
            return ((prediction_arguments == target_arguments).float()).mean()



if __name__ == "__main__":
    raise Exception(
        "network.py - test of basic functions not yet implemented.")
