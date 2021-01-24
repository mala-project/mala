import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import warnings
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    warnings.warn("You either don't have Horovod installed or it is not configured correctly. You can still "
              "train networks, but attempting to set parameters.training.use_horovod = True WILL cause a crash.")


class Trainer:
    """A class for training a neural network."""

    def __init__(self, p):
        # copy the parameters into the class.
        self.parameters = p.training
        self.final_test_loss = float("inf")
        self.optimizer = None
        self.scheduler = None
        self.network = None
        self.batch_size=p.training.mini_batch_size
        self.use_gpu = False
        self.use_horovod=False
        

    def train_network(self, network, data):
        """Given a network and data, this network is trained on this data."""

        # See if we can and want to work on a GPU.
        self.use_gpu = torch.cuda.is_available() and self.parameters.use_gpu

        # This is a place where additional checks could be placed.
        self.use_horovod= self.parameters.use_horovod

        # See if we want to use horovod.
        if self.use_horovod:
            # Initialize horovod
            hvd.init()
            if self.use_gpu:
                print("size=", hvd.size(), "global_rank=", hvd.rank(), "local_rank=", hvd.local_rank(), "device=",
                      torch.cuda.get_device_name(hvd.local_rank()))
                # pin GPU to local rank
                torch.cuda.set_device(hvd.local_rank())

            # Using seeds to repreduce the same result
            torch.manual_seed(self.parameters.seed)
            if self.use_gpu:
                torch.cuda.manual_seed(self.parameters.seed)

  

        # If we choose to work on a GPU, we need to move the network to this GPU.
        if self.use_gpu:
            network.to('cuda')

        # Scale the learning rate according to horovod. 
        if self.use_horovod:
                if hvd.size() > 1:
                        print("Rescaling learning rate because multiple workers are used for training.")
                        self.parameters.learning_rate = self.parameters.learning_rate * hvd.size() 

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
            self.parameters.kwargs = {'num_workers': 0, 'pin_memory': True}

            #Set the data sampler for multiGPU
            self.parameters.sampler["train_sampler"] = torch.utils.data.distributed.DistributedSampler(data.training_data_set,
                                                                                                       num_replicas=hvd.size(),
                                                                                                       rank=hvd.rank())

            self.parameters.sampler["validate_sampler"] = torch.utils.data.distributed.DistributedSampler(data.validation_data_set,
                                                                                                          num_replicas=hvd.size(),
                                                                                                          rank=hvd.rank())

            self.parameters.sampler["test_sampler"] =torch.utils.data.distributed.DistributedSampler(data.test_data_set,
                                                                                                     num_replicas=hvd.size(),
                                                                                                     rank=hvd.rank())

            # broadcaste parameters and optimizer state from root device to other devices
            hvd.broadcast_parameters(network.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

            # Wraps the opimizer for multiGPU operation
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,  
                                                      named_parameters=network.named_parameters(),
                                                      op = hvd.Average)


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
        training_data_loader = DataLoader(data.training_data_set, batch_size=self.batch_size,
                                          sampler=self.parameters.sampler["train_sampler"],
                                          **self.parameters.kwargs ) #shuffle=True,

        validation_data_loader = DataLoader(data.validation_data_set, batch_size=self.batch_size * 1,
                                            sampler=self.parameters.sampler["validate_sampler"],**self.parameters.kwargs )

        test_data_loader = DataLoader(data.test_data_set, batch_size=self.batch_size * 1,
                                      sampler=self.parameters.sampler["test_sampler"],**self.parameters.kwargs )


        # Calculate initial loss.
        vloss = self.validate_network(network, validation_data_loader)
        tloss = self.validate_network(network, test_data_loader)

        #Collect and average all the losses from all the devices
        if self.use_horovod:
            print(hvd.rank(), vloss)
            vloss=self.average_validation(vloss,'average_loss')
            tloss=self.average_validation(tloss,'average_loss')
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
            if self.use_horovod:
                vloss=self.average_validation(vloss,'average_loss')
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
        if self.use_horovod:
            tloss=self.average_validation(tloss,'average_loss')
        self.final_test_loss = tloss
        print("Final test data loss: ", tloss)

    def process_mini_batch(self, network, input_data, target_data):
        prediction = network.forward(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def validate_network(self, network, vdl):
        network.eval()
        validation_loss = []
        with torch.no_grad():
            for x, y in vdl:
                if self.use_gpu:
                    x = x.to('cuda')
                    y = y.to('cuda')
                prediction = network(x)
                validation_loss.append(network.calculate_loss(prediction, y).item())
        return np.mean(validation_loss)

    def average_validation(self,val,name):
        tensor= torch.tensor(val)
        avg_loss=hvd.allreduce(tensor,name=name, op=hvd.Average)
        return avg_loss.item()
