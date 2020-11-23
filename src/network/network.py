'''
network.py contains the neural network class used in this framework.
'''

import torch.nn as nn
import torch.nn.functional as F



class network(nn.Module):
    """Central network class for this framework. Based on pytorch.nn.Module."""
    def __init__(self, p):
        # copy the network parameters from the input parameter object
        self.parameters = p.network
        # initialize the parent class
        super(network, self).__init__()

        # initialize the layers
        self.number_of_layers = len(self.parameters.layer_sizes)
        self.layers = []
        if (self.parameters.nn_type == "feed-forward"):
            self.initialize_as_feedforward()
        else:
            raise Exception("Unsupported network architecture.")

        # initialize the loss function
        if (self.parameters.loss_function_type == "cross_entropy"):
            self.loss_func = F.cross_entropy
        else:
            raise Exception("Unsupported loss function.")


    def initialize_as_feedforward(self):
        for i in range(0,self.number_of_layers-1):
            self.layers.append((nn.Linear(self.parameters.layer_sizes[i], self.parameters.layer_sizes[i+1])))


    def forward(self, input):
        if (self.parameters.nn_type == "feed-forward"):
            for i in range(0,self.number_of_layers-1):
                input = self.layers[i](input)
            return input
        else:
            raise Exception("Unsupported network architecture.")

    def calculate_loss(self, input, output):
        return self.loss_func(input, target)



if __name__ == "__main__":
    raise Exception(
        "network.py - test of basic functions not yet implemented.")
