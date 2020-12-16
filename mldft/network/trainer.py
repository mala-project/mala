import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader


class Trainer:
    """A class for training a neural network."""

    def __init__(self, p):
        # copy the parameters into the class.
        self.parameters = p.training
        self.final_test_loss = float("inf")

    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            optimizer = optim.SGD(network.parameters(), lr=self.parameters.learning_rate)
        elif self.parameters.trainingtype == "Adam":
            optimizer = optim.Adam(network.parameters(), lr=self.parameters.learning_rate)
        else:
            raise Exception("Unsupported training method.")

        # Prepare data loaders.
        training_data_loader = DataLoader(data.training_data_set, batch_size=self.parameters.mini_batch_size,
                                          shuffle=True)
        validation_data_loader = DataLoader(data.validation_data_set, batch_size=self.parameters.mini_batch_size * 1)
        test_data_loader = DataLoader(data.test_data_set, batch_size=self.parameters.mini_batch_size * 1)

        # Calculate initial loss.
        if self.parameters.verbosity == True:
            vloss = self.validate_network(network, validation_data_loader)
            print("Initial Guess - validation data loss: ", vloss)
            tloss = self.validate_network(network, test_data_loader)
            print("Initial Guess - test data loss: ", tloss)

        # Perform and log training.
        for epoch in range(self.parameters.max_number_epochs):
            network.train()
            for inputs, outputs in training_data_loader:
                self.process_mini_batch(network, optimizer, inputs, outputs)
            if self.parameters.verbosity == True:
                vloss = self.validate_network(network, validation_data_loader)
                print("Epoch: ", epoch, "validation data loss: ", vloss)

        # Calculate final loss.
        tloss = self.validate_network(network, test_data_loader)
        self.final_test_loss = tloss
        print("Final test data loss: ", tloss)

    @staticmethod
    def process_mini_batch(network, optimizer, input_data, target_data):
        prediction = network.forward(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # FIXME: This seems inefficient.
    @staticmethod
    def validate_network(network, vdl):
        network.eval()
        accuracies = []
        validation_loss = 0
        with torch.no_grad():
            for (x, y) in vdl:
                prediction = network(x)
                validation_loss += network.loss_func(prediction, y).item()
                # accuracies.append(network.classification_accuracy(prediction, y))
            # validation_accuracy = np.mean(accuracies)
        return validation_loss
