"""Trainer class for training a network."""
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from fesl.common.parameters import printout
from .runner import Runner
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import time


class Trainer(Runner):
    """A class for training a neural network."""

    def __init__(self, params):
        """
        Create a Trainer object to run a Network.

        Parameters
        ----------
        params : fesl.common.parametes.Parameters
            Parameters used to create this Trainer object.
        """
        # copy the parameters into the class.
        super(Trainer, self).__init__(params)
        self.final_test_loss = float("inf")
        self.initial_test_loss = float("inf")
        self.optimizer = None
        self.scheduler = None

    def train_network(self, network, data):
        """
        Train a network using data given by a DataHandler.

        Parameters
        ----------
        network : fesl.network.network.Network
            Network which is being trained.

        data : fesl.datahandling.data_handler.DataHandler
            DataHandler holding the training data.
        """
        # Prepare horovod to run.
        super(Trainer, self).prepare_to_run()

        # Configure keyword arguments for DataSampler.
        if self.use_gpu:
            self.parameters.kwargs['pin_memory'] = True

        # Scale the learning rate according to horovod. 
        if self.use_horovod:
            if hvd.size() > 1:
                printout("Rescaling learning rate because multiple workers are"
                         " used for training.")
                self.parameters.learning_rate = self.parameters.learning_rate \
                    * hvd.size()

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            self.optimizer = optim.SGD(network.parameters(),
                                       lr=self.parameters.learning_rate,
                                       weight_decay=self.parameters.
                                       weight_decay)

        elif self.parameters.trainingtype == "Adam":
            self.optimizer = optim.Adam(network.parameters(),
                                        lr=self.parameters.learning_rate,
                                        weight_decay=self.parameters.
                                        weight_decay)
        else:
            raise Exception("Unsupported training method.")

        if self.use_horovod:
            # scaling the batch size for multiGPU per node
            # self.batch_size= self.batch_size*hvd.local_size()

            compression = hvd.Compression.fp16 if self.use_compression \
                else hvd.Compression.none

            # If lazy loading is used we do not shuffle the data points on
            # their own, but rather shuffle them
            # by shuffling the files themselves and then reading file by file
            # per epoch.
            # This shuffling is done in the dataset themselves.
            do_shuffle = self.parameters.use_shuffling_for_samplers
            if data.parameters.use_lazy_loading:
                do_shuffle = False

            # Set the data sampler for multiGPU
            self.parameters.sampler["train_sampler"] = torch.utils.data.\
                distributed.DistributedSampler(data.training_data_set,
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=do_shuffle)

            self.parameters.sampler["validate_sampler"] = torch.utils.data.\
                distributed.DistributedSampler(data.validation_data_set,
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=False)

            if data.test_data_set is not None:
                self.parameters.sampler["test_sampler"] = torch.utils.data.\
                    distributed.DistributedSampler(data.test_data_set,
                                                   num_replicas=hvd.size(),
                                                   rank=hvd.rank(),
                                                   shuffle=False)

            # broadcaste parameters and optimizer state from root device to
            # other devices
            hvd.broadcast_parameters(network.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

            # Wraps the opimizer for multiGPU operation
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,  
                                                      named_parameters=
                                                      network.
                                                      named_parameters(),
                                                      compression=compression,
                                                      op=hvd.Average)

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.\
                lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                               patience=self.parameters.
                                               learning_rate_patience,
                                               mode="min",
                                               factor=self.parameters.
                                               learning_rate_decay,
                                               verbose=True)
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate schedule.")

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        test_data_loader = None
        if data.parameters.use_lazy_loading or self.use_horovod:
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        training_data_loader = DataLoader(data.training_data_set,
                                          batch_size=self.batch_size,
                                          sampler=self.parameters.
                                          sampler["train_sampler"],
                                          **self.parameters.kwargs,
                                          shuffle=do_shuffle)

        validation_data_loader = DataLoader(data.validation_data_set,
                                            batch_size=self.batch_size * 1,
                                            sampler=self.parameters.
                                            sampler["validate_sampler"],
                                            **self.parameters.kwargs)

        if data.test_data_set is not None:
            test_data_loader = DataLoader(data.test_data_set,
                                          batch_size=self.batch_size * 1,
                                          sampler=self.parameters.
                                          sampler["test_sampler"],
                                          **self.parameters.kwargs)

        # Calculate initial loss.
        tloss = None
        vloss = self.__validate_network(network, validation_data_loader)
        if data.test_data_set is not None:
            tloss = self.__validate_network(network, test_data_loader)

        # Collect and average all the losses from all the devices
        if self.use_horovod:
            vloss = self.__average_validation(vloss, 'average_loss')
            if data.test_data_set is not None:
                tloss = self.__average_validation(tloss, 'average_loss')
        if self.parameters.verbosity:
            printout("Initial Guess - validation data loss: ", vloss)
            if data.test_data_set is not None:
                printout("Initial Guess - test data loss: ", tloss)
        self.initial_test_loss = tloss

        # Perform and log training.
        patience_counter = 0
        vloss_old = vloss
        for epoch in range(self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            network.train()

            # Process each mini batch and save the training loss.
            training_loss = 0
            # train sampler 
            if self.use_horovod:
                self.parameters.sampler["train_sampler"].set_epoch(epoch)

            # nr_of_batches = (data.nr_training_data // self.batch_size)+1
            # oldprogress = 0

            for batchid, (inputs, outputs) in enumerate(training_data_loader):
                if self.use_gpu:

                    inputs = inputs.to('cuda')
                    outputs = outputs.to('cuda')
                training_loss += self.__process_mini_batch(network, inputs,
                                                           outputs)

                # Output of the progress, this can be useful for tests if
                # an epoch takes forever and one is not
                # sure if something went wrong or everythin is simply taking
                # a while.
                # progress = 100*batchid/nr_of_batches
                # if progress - oldprogress > 10:
                #     printout("{0:10.2f} % of epoch finished".
                #     format(progress))
                #     oldprogress = progress

            # Calculate the validation loss. and output it.
            vloss = self.__validate_network(network, validation_data_loader)
            if self.use_horovod:
                vloss = self.__average_validation(vloss, 'average_loss')
            if self.parameters.verbosity:
                printout("Epoch: ", epoch, "validation data loss: ", vloss)

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                data.mix_datasets()

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if self.parameters.learning_rate_scheduler ==\
                        "ReduceLROnPlateau":
                    self.scheduler.step(vloss)
            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (1.0 + self.parameters.
                                        early_stopping_threshold):
                    patience_counter = 0
                    vloss_old = vloss
                else:
                    patience_counter += 1
                    printout("Validation accuracy has not improved enough.")
                    if patience_counter >= self.parameters.\
                            early_stopping_epochs:
                        if self.parameters.verbosity:
                            printout("Stopping the training, validation "
                                     "accuracy has not improved for",
                                     patience_counter,
                                     "epochs.")
                        break
            printout("Time for epoch[s]:", time.time() - start_time)

        # Calculate final loss.
        tloss = None
        if data.test_data_set is not None:
            tloss = self.__validate_network(network, test_data_loader)
            if self.use_horovod:
                tloss = self.__average_validation(tloss, 'average_loss')
        self.final_test_loss = tloss
        printout("Final test data loss: ", tloss)

    def __process_mini_batch(self, network, input_data, target_data):
        """Process a mini batch."""
        prediction = network.forward(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def __validate_network(self, network, vdl):
        """Validate a network, using test or validation data."""
        network.eval()
        validation_loss = []
        with torch.no_grad():
            for x, y in vdl:
                if self.use_gpu:
                    x = x.to('cuda')
                    y = y.to('cuda')
                prediction = network(x)
                validation_loss.append(network.calculate_loss(prediction, y)
                                       .item())

        return np.mean(validation_loss)

    @staticmethod
    def __average_validation(val, name):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val)
        avg_loss = hvd.allreduce(tensor, name=name, op=hvd.Average)
        return avg_loss.item()
