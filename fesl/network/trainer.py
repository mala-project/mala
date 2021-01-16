import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import horovod.torch as hvd

class Trainer:
    """A class for training a neural network."""

    def __init__(self, p):
        # copy the parameters into the class.
        self.parameters = p.training
        self.final_test_loss = float("inf")
        self.optimizer = None
        self.scheduler = None
        self.network = None
        self.use_gpu = False
        self.use_horovod=False
        

    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""

        # See if we can and want to work on a GPU.
        self.use_gpu = torch.cuda.is_available() and self.parameters.use_gpu
        #see if we want to use hor9ovod for multi GPU.
        if torch.cuda.device_count() == 1: # change to >1
            self.use_horovod= self.use_gpu and self.parameters.use_horovod
        if self.use_horovod:
            # Initialize horovod
            hvd.init()
            # pin GPU to local rank
            torch.cuda.set_device(hvd.local_rank())

            # Using seeds to repreduce the same result
            torch.manual_seed(self.parameters.seed)
            torch.cuda.manual_seed(self.parameters.seed)
           
  

        # If we choose to work on a GPU, we need to move the network to this GPU.
        if self.use_gpu:
            network.to('cuda')

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            self.optimizer = optim.SGD(network.parameters(), lr=self.parameters.learning_rate,
                                       weight_decay=self.parameters.weight_decay)

        elif self.parameters.trainingtype == "Adam":
            self.optimizer = optim.Adam(network.parameters(), lr=self.parameters.learning_rate,
                                        weight_decay=self.parameters.weight_decay)
                                        

        else:
            raise Exception("Unsupported training method.")

        if self.use_horovod:
            self.parameters.kwargs = {'num_workers': 1, 'pin_memory': True}
            self.parameters.sampler["train_sampler"] = torch.utils.data.distributed.DistributedSampler(data.training_data_set, 
                                                                                                       num_replicas=hvd.size(), 
                                                                                                       rank=hvd.rank())
            # add num_replicas and rank to the class)
            self.parameters.sampler["validate_sampler"] = torch.utils.data.distributed.DistributedSampler(data.validation_data_set, 
                                                                                                          num_replicas=hvd.size(), 
                                                                                                          rank=hvd.rank())
            # 
            self.parameters.sampler["test_sampler"] =torch.utils.data.distributed.DistributedSampler(data.test_data_set, 
                                                                                                     num_replicas=hvd.size(), 
                                                                                                     rank=hvd.rank())
            # broadcasting
            hvd.broadcast_parameters(network.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            # add comment
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,  
                                                      named_parameters=network.named_parameters(),
                                                      op = hvd.Adasum)

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

        # Prepare data loaders.(look into mini-batch size)
        training_data_loader = DataLoader(data.training_data_set, batch_size=self.parameters.mini_batch_size,
                                          sampler=self.parameters.sampler["train_sampler"],
                                          **self.parameters.kwargs ) #shuffle=True,
        validation_data_loader = DataLoader(data.validation_data_set, batch_size=self.parameters.mini_batch_size * 1,
                                            sampler=self.parameters.sampler["validate_sampler"],**self.parameters.kwargs )
        test_data_loader = DataLoader(data.test_data_set, batch_size=self.parameters.mini_batch_size * 1,
                                      sampler=self.parameters.sampler["test_sampler"],**self.parameters.kwargs )

        # Calculate initial loss.
        # add here the all reduce
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
            # train sampler 
        
            self.parameters.sampler["train_sampler"].set_epoch(epoch)
            for inputs, outputs in training_data_loader:
                if self.use_gpu:
                    inputs = inputs.to('cuda')
                    outputs = outputs.to('cuda')
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
    def validate_network(self, network, vdl):
        network.eval()
        accuracies = []
        validation_loss = 0
        with torch.no_grad():
            for x, y in vdl:
                if self.use_gpu:
                    x = x.to('cuda')
                    y = y.to('cuda')
                prediction = network(x)
                validation_loss += network.calculate_loss(prediction, y).item()
                # accuracies.append(network.classification_accuracy(prediction, y))
            # validation_accuracy = np.mean(accuracies)
        return validation_loss
