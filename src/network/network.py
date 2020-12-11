import torch
import torch.nn as nn
import torch.nn.functional as F




class Network(nn.Module):
    """Central network class for this framework. Based on pytorch.nn.Module."""
    def __init__(self, p):
        # copy the network params from the input parameter object
        self.params = p.Network
        # initialize the parent class
        super(Network, self).__init__()

        # Mappings for parsing of the activation layers.
        self.activation_mappings = {
            "Sigmoid" : nn.Sigmoid,
            "ReLU" : nn.ReLU,
        }

        # initialize the layers
        self.number_of_layers = len(self.params.layer_sizes)-1
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
        # Check if multiple types of activations were selected or only one was passed to be used in the entire network.
        use_only_one_activation_type = False
        if (len(self.params.layer_activations) == 1):
            use_only_one_activation_type = True
        elif(len(self.params.layer_activations) < self.number_of_layers):
            raise Exception("Not enough activation layers provided.")
        for i in range(0,self.number_of_layers):
            self.layers.append((nn.Linear(self.params.layer_sizes[i], self.params.layer_sizes[i+1])))
            try:
                if (use_only_one_activation_type):
                    self.layers.append(self.activation_mappings[self.params.layer_activations[0]]())
                else:
                    self.layers.append(self.activation_mappings[self.params.layer_activations[i]]())
            except:
                raise Exception("Invalid activation type seleceted.")


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

    @staticmethod
    def classification_accuracy(prediction, target):
            prediction_arguments = torch.argmax(prediction, dim=1)
            target_arguments = torch.argmax(target, dim=1)
            return ((prediction_arguments == target_arguments).float()).mean()
