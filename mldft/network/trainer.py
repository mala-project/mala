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
        self.optimizer = None
        self.scheduler = None
        self.network = None

    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            self.optimizer = optim.SGD(network.parameters(), lr=self.parameters.learning_rate)
        elif self.parameters.trainingtype == "Adam":
            self.optimizer = optim.Adam(network.parameters(), lr=self.parameters.learning_rate)
        else:
            raise Exception("Unsupported training method.")

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  patience=self.parameters.learning_rate_patience,
                                                                  mode="min",
                                                                  factor=self.parameters.learning_rate_decay,
                                                                  verbose=True)
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate schedule.")

        # Prepare data loaders.
        training_data_loader = DataLoader(data.training_data_set, batch_size=self.parameters.mini_batch_size,
                                          shuffle=True)
        validation_data_loader = DataLoader(data.validation_data_set, batch_size=self.parameters.mini_batch_size * 1)
        test_data_loader = DataLoader(data.test_data_set, batch_size=self.parameters.mini_batch_size * 1)

        # Calculate initial loss.
        vloss = self.validate_network(network, validation_data_loader)
        tloss = self.validate_network(network, test_data_loader)
        if self.parameters.verbosity:
            print("Initial Guess - validation data loss: ", vloss)
            print("Initial Guess - test data loss: ", tloss)

        # Perform and log training.
        patience_counter = 0
        vloss_old = vloss
        for epoch in range(self.parameters.max_number_epochs):

            # Prepare model for training.
            network.train()

            # Process each mini batch and save the training loss.
            training_loss = 0
            for inputs, outputs in training_data_loader:
                training_loss += self.process_mini_batch(network, inputs, outputs)

            # Calculate the validation loss.
            vloss = self.validate_network(network, validation_data_loader)
            if self.parameters.verbosity:
                print("Epoch: ", epoch, "validation data loss: ", vloss)

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (1.0 + self.parameters.early_stopping_threshold):
                    patience_counter = 0
                    vloss_old = vloss
                else:
                    patience_counter += 1
                    print("Validation accuracy has not improved enough.")
                    if patience_counter >= self.parameters.early_stopping_epochs:
                        if self.parameters.verbosity:
                            print("Stopping the training, validation accuracy has not improved for", patience_counter,
                                  "epochs.")
                        break

        # Calculate final loss.
        tloss = self.validate_network(network, test_data_loader)
        self.final_test_loss = tloss
        print("Final test data loss: ", tloss)

    def process_mini_batch(self, network, input_data, target_data):
        prediction = network.forward(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # FIXME: This seems inefficient.
    @staticmethod
    def validate_network(network, vdl):
        network.eval()
        accuracies = []
        validation_loss = 0
        with torch.no_grad():
            for (x, y) in vdl:
                prediction = network(x)
                validation_loss += network.calculate_loss(prediction, y).item()
                # accuracies.append(network.classification_accuracy(prediction, y))
            # validation_accuracy = np.mean(accuracies)
        return validation_loss
