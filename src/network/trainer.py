'''
trainer.py contains the training class used to train a network.
This could be absorbed into the network.py class and maybe it will be at some point
or serve as a parent class for multiple trainers, should need be.
'''

import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

class trainer():
    """A class for training a neural network."""

    def __init__(self, p):
        # copy the parameters into the class.
        self.parameters = p.training


    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""
        if (self.parameters.trainingtype == "SGD"):
            optimizer = optim.SGD(network.parameters(), lr=self.parameters.learning_rate)
        else:
            raise Exception("Unsupported training method.")
        training_data_loader = DataLoader(data.training_data_set, batch_size = self.parameters.mini_batch_size, shuffle=True)
        validation_data_loader = DataLoader(data.validation_data_set, batch_size = self.parameters.mini_batch_size*1)

        if (self.parameters.verbosity == True):
            vloss, vaccuracy = self.validate_network(network, validation_data_loader)
            print("Initial Guess - Validation Loss: " ,vloss,"Validation accuracy: ",vaccuracy)

        for epoch in range(self.parameters.max_number_epochs):
            network.train()
            for input, output in training_data_loader:
                self.process_mini_batch(network, optimizer, input, output)
            if (self.parameters.verbosity == True):
                vloss, vaccuracy = self.validate_network(network, validation_data_loader)
                print("Epoch: ",epoch,"Validation Loss: " ,vloss,"Validation accuracy: ",vaccuracy)

    def process_mini_batch(self, network, optimizer, input_data, target_data):
        prediction = network.forward(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # FIXME: This seems inefficient.
    def validate_network(self, network, vdl):
        network.eval()
        accuracies = []
        validation_loss = 0
        with torch.no_grad():
            for (x,y) in vdl:
                prediction = network(x)
                validation_loss += network.loss_func(prediction, y).item()
                accuracies.append(network.classification_accuracy(prediction, y))
            validation_accuracy = np.mean(accuracies)
        return validation_loss, validation_accuracy


if __name__ == "__main__":
    raise Exception(
        "trainer.py - test of basic functions not yet implemented.")
