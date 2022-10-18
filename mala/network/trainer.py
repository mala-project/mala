"""Trainer class for training a network."""
import os
import time
from datetime import datetime

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mala.common.parameters import Parameters
from mala.common.parameters import printout
from mala.datahandling.data_handler import DataHandler
from mala.datahandling.data_scaler import DataScaler
from mala.network.network import Network
from mala.network.runner import Runner


class Trainer(Runner):
    """A class for training a neural network.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Trainer object.

    network : mala.network.network.Network
        Network which is being trained.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the training data.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, network, data, optimizer_dict=None,
                 use_pkl_checkpoints=False):
        # copy the parameters into the class.
        super(Trainer, self).__init__(params, network, data)
        self.final_test_loss = float("inf")
        self.initial_test_loss = float("inf")
        self.final_validation_loss = float("inf")
        self.initial_validation_loss = float("inf")
        self.optimizer = None
        self.scheduler = None
        self.patience_counter = 0
        self.last_epoch = 0
        self.last_loss = None
        self.training_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None
        self.use_pkl_checkpoints = use_pkl_checkpoints

        # Samplers for the horovod case.
        self.train_sampler = None
        self.test_sampler = None
        self.validation_sampler = None

        self.__prepare_to_train(optimizer_dict)

        self.tensor_board = None
        self.full_visualization_path = None
        if self.parameters.visualisation:
            if not os.path.exists(self.parameters.visualisation_dir):
                os.makedirs(self.parameters.visualisation_dir)
            if self.parameters.visualisation_dir_append_date:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.full_visualization_path = \
                    os.path.join(self.parameters.visualisation_dir, date_time)
                os.makedirs(self.full_visualization_path)
            else:
                self.full_visualization_path = \
                    self.parameters.visualisation_dir

            # Set the path to log files
            self.tensor_board = SummaryWriter(self.full_visualization_path)
            printout("Writing visualization output to",
                     self.full_visualization_path, min_verbosity=1)

    @classmethod
    def checkpoint_exists(cls, checkpoint_name, use_pkl_checkpoints=False):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        network_name = checkpoint_name + "_network.pth"
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        if use_pkl_checkpoints:
            param_name = checkpoint_name + "_params.pkl"
        else:
            param_name = checkpoint_name + "_params.json"
        optimizer_name = checkpoint_name + "_optimizer.pth"

        return all(map(os.path.isfile, [iscaler_name, oscaler_name, param_name,
                                        network_name, optimizer_name]))

    @classmethod
    def resume_checkpoint(cls, checkpoint_name, use_pkl_checkpoints=False):
        """
        Prepare resumption of training from a checkpoint.

        Please note that to actually resume the training,
        Trainer.train_network() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved in the checkpoint.

        loaded_network : mala.network.network.Network
            The network saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_trainer : Trainer
            The trainer reconstructed from the checkpoint.
        """
        printout("Loading training run from checkpoint.", min_verbosity=0)
        # The names are based upon the checkpoint name.
        network_name = checkpoint_name + "_network.pth"
        iscaler_name = checkpoint_name + "_iscaler.pkl"
        oscaler_name = checkpoint_name + "_oscaler.pkl"
        if use_pkl_checkpoints:
            param_name = checkpoint_name + "_params.pkl"
        else:
            param_name = checkpoint_name + "_params.json"
        optimizer_name = checkpoint_name + "_optimizer.pth"

        # First load the all the regular objects.
        loaded_params = Parameters.load_from_file(param_name)
        loaded_iscaler = DataScaler.load_from_file(iscaler_name)
        loaded_oscaler = DataScaler.load_from_file(oscaler_name)
        loaded_network = Network.load_from_file(loaded_params,
                                                network_name)

        printout("Preparing data used for last checkpoint.", min_verbosity=0)
        # Create a new data handler and prepare the data.
        new_datahandler = DataHandler(loaded_params,
                                      input_data_scaler=loaded_iscaler,
                                      output_data_scaler=loaded_oscaler)
        new_datahandler.prepare_data(reparametrize_scaler=False)
        new_trainer = Trainer.load_from_file(loaded_params, optimizer_name,
                                             loaded_network, new_datahandler)

        return loaded_params, loaded_network, new_datahandler, new_trainer

    @classmethod
    def load_from_file(cls, params, file_path, network, data):
        """
        Load a trainer from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the trainer should be created.
            Has to be compatible with network and data.

        file_path : string
            Path to the file from which the trainer should be loaded.

        network : mala.network.network.Network
            Network which is being trained.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.


        Returns
        -------
        loaded_trainer : Network
            The trainer that was loaded from the file.
        """
        # First, load the checkpoint.
        checkpoint = torch.load(file_path)

        # Now, create the Trainer class with it.
        loaded_trainer = Trainer(params, network, data,
                                 optimizer_dict=checkpoint)
        return loaded_trainer

    def train_network(self):
        """Train a network using data given by a DataHandler."""
        ############################
        # CALCULATE INITIAL METRICS
        ############################

        tloss = float("inf")
        vloss = self.__validate_network(self.network,
                                        "validation",
                                        self.parameters.
                                        after_before_training_metric)

        if self.data.test_data_set is not None:
            tloss = self.__validate_network(self.network,
                                            "test",
                                            self.parameters.
                                            after_before_training_metric)

        # Collect and average all the losses from all the devices
        if self.parameters_full.use_horovod:
            vloss = self.__average_validation(vloss, 'average_loss')
            self.initial_validation_loss = vloss
            if self.data.test_data_set is not None:
                tloss = self.__average_validation(tloss, 'average_loss')
                self.initial_test_loss = tloss

        printout("Initial Guess - validation data loss: ", vloss,
                 min_verbosity=1)
        if self.data.test_data_set is not None:
            printout("Initial Guess - test data loss: ", tloss,
                     min_verbosity=1)

        # Save losses for later use.
        self.initial_validation_loss = vloss
        self.initial_test_loss = tloss

        # Initialize all the counters.
        checkpoint_counter = 0

        # If we restarted from a checkpoint, we have to differently initialize
        # the loss.
        if self.last_loss is None:
            vloss_old = vloss
        else:
            vloss_old = self.last_loss

        ############################
        # PERFORM TRAINING
        ############################

        for epoch in range(self.last_epoch, self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            self.network.train()

            # Process each mini batch and save the training loss.
            training_loss = []

            # train sampler
            if self.parameters_full.use_horovod:
                self.train_sampler.set_epoch(epoch)

            for batchid, (inputs, outputs) in \
                    enumerate(self.training_data_loader):
                inputs = inputs.to(self.parameters._configuration["device"])
                outputs = outputs.to(self.parameters._configuration["device"])
                training_loss.append(self.__process_mini_batch(self.network,
                                                               inputs,
                                                               outputs))
            training_loss = np.mean(training_loss)

            # Calculate the validation loss. and output it.
            vloss = self.__validate_network(self.network,
                                            "validation",
                                            self.parameters.
                                            during_training_metric)

            if self.parameters_full.use_horovod:
                vloss = self.__average_validation(vloss, 'average_loss')
            if self.parameters_full.verbosity > 1:
                printout("Epoch {0}: validation data loss: {1}, "
                         "training data loss: {2}".format(epoch, vloss,
                                                          training_loss),
                         min_verbosity=2)
            else:
                printout("Epoch {0}: validation data loss: {1}".format(epoch,
                                                                       vloss),
                         min_verbosity=1)

            # summary_writer tensor board
            if self.parameters.visualisation:
                self.tensor_board.add_scalars('Loss', {'validation': vloss,
                                              'training': training_loss},
                                              epoch)
                self.tensor_board.add_scalar("Learning rate",
                                             self.parameters.learning_rate,
                                             epoch)
                if self.parameters.visualisation == 2:
                    for name, param in self.network.named_parameters():
                        self.tensor_board.add_histogram(name, param, epoch)
                        self.tensor_board.add_histogram(f'{name}.grad',
                                                        param.grad, epoch)

                # method to make sure that all pending events have been written
                # to disk
                self.tensor_board.close()

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                self.data.mix_datasets()

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if self.parameters.learning_rate_scheduler ==\
                        "ReduceLROnPlateau":
                    self.scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (1.0 + self.parameters.
                                        early_stopping_threshold):
                    self.patience_counter = 0
                    vloss_old = vloss
                else:
                    self.patience_counter += 1
                    printout("Validation accuracy has not improved "
                             "enough.", min_verbosity=1)
                    if self.patience_counter >= self.parameters.\
                            early_stopping_epochs:
                        printout("Stopping the training, validation "
                                 "accuracy has not improved for",
                                 self.patience_counter,
                                 "epochs.", min_verbosity=1)
                        self.last_epoch = epoch
                        break

            # If checkpointing is enabled, we need to checkpoint.
            if self.parameters.checkpoints_each_epoch != 0:
                checkpoint_counter += 1
                if checkpoint_counter >= \
                        self.parameters.checkpoints_each_epoch:
                    printout("Checkpointing training.", min_verbosity=0)
                    self.last_epoch = epoch
                    self.last_loss = vloss_old
                    self.__create_training_checkpoint()
                    checkpoint_counter = 0

            printout("Time for epoch[s]:", time.time() - start_time,
                     min_verbosity=2)

        ############################
        # CALCULATE FINAL METRICS
        ############################

        if self.parameters.after_before_training_metric != \
                self.parameters.during_training_metric:
            vloss = self.__validate_network(self.network,
                                            "validation",
                                            self.parameters.
                                            after_before_training_metric)
            if self.parameters_full.use_horovod:
                vloss = self.__average_validation(vloss, 'average_loss')

        # Calculate final loss.
        self.final_validation_loss = vloss
        printout("Final validation data loss: ", vloss, min_verbosity=0)

        tloss = float("inf")
        if self.data.test_data_set is not None:
            tloss = self.__validate_network(self.network,
                                            "test",
                                            self.parameters.
                                            after_before_training_metric)
            if self.parameters_full.use_horovod:
                tloss = self.__average_validation(tloss, 'average_loss')
            printout("Final test data loss: ", tloss, min_verbosity=0)
        self.final_test_loss = tloss

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {'num_workers': self.parameters.num_workers,
                  'pin_memory': False}
        if self.parameters_full.use_gpu:
            kwargs['pin_memory'] = True

        # Scale the learning rate according to horovod.
        if self.parameters_full.use_horovod:
            if hvd.size() > 1:
                printout("Rescaling learning rate because multiple workers are"
                         " used for training.", min_verbosity=1)
                self.parameters.learning_rate = self.parameters.learning_rate \
                    * hvd.size()

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.parameters.learning_rate,
                                       weight_decay=self.parameters.
                                       weight_decay)
        elif self.parameters.trainingtype == "Adam":
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=self.parameters.learning_rate,
                                        weight_decay=self.parameters.
                                        weight_decay)
        else:
            raise Exception("Unsupported training method.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            self.optimizer.\
                load_state_dict(optimizer_dict['optimizer_state_dict'])
            self.last_epoch = optimizer_dict['epoch']+1
            self.patience_counter = optimizer_dict['early_stopping_counter']
            self.last_loss = optimizer_dict['early_stopping_last_loss']

        if self.parameters_full.use_horovod:
            # scaling the batch size for multiGPU per node
            # self.batch_size= self.batch_size*hvd.local_size()

            compression = hvd.Compression.fp16 if self.parameters_full.\
                running.use_compression else hvd.Compression.none

            # If lazy loading is used we do not shuffle the data points on
            # their own, but rather shuffle them
            # by shuffling the files themselves and then reading file by file
            # per epoch.
            # This shuffling is done in the dataset themselves.
            do_shuffle = self.parameters.use_shuffling_for_samplers
            if self.data.parameters.use_lazy_loading:
                do_shuffle = False

            self.train_sampler = torch.utils.data.\
                distributed.DistributedSampler(self.data.training_data_set,
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=do_shuffle)

            self.validation_sampler = torch.utils.data.\
                distributed.DistributedSampler(self.data.validation_data_set,
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=False)

            if self.data.test_data_set is not None:
                self.test_sampler = torch.utils.data.\
                    distributed.DistributedSampler(self.data.test_data_set,
                                                   num_replicas=hvd.size(),
                                                   rank=hvd.rank(),
                                                   shuffle=False)

            # broadcaste parameters and optimizer state from root device to
            # other devices
            hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

            # Wraps the opimizer for multiGPU operation
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,
                                                      named_parameters=
                                                      self.network.
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
        if self.scheduler is not None and optimizer_dict is not None:
            self.scheduler.\
                load_state_dict(optimizer_dict['lr_scheduler_state_dict'])

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        if self.data.parameters.use_lazy_loading or self.parameters_full.\
                use_horovod:
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        self.training_data_loader = DataLoader(self.data.training_data_set,
                                               batch_size=self.parameters.
                                               mini_batch_size,
                                               sampler=self.train_sampler,
                                               **kwargs,
                                               shuffle=do_shuffle)

        self.validation_data_loader = DataLoader(self.data.validation_data_set,
                                                 batch_size=self.parameters.
                                                 mini_batch_size * 1,
                                                 sampler=
                                                 self.validation_sampler,
                                                 **kwargs)

        if self.data.test_data_set is not None:
            self.test_data_loader = DataLoader(self.data.test_data_set,
                                               batch_size=self.parameters.
                                               mini_batch_size * 1,
                                               sampler=self.test_sampler,
                                               **kwargs)

    def __process_mini_batch(self, network, input_data, target_data):
        """Process a mini batch."""
        prediction = network(input_data)
        loss = network.calculate_loss(prediction, target_data)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def __validate_network(self, network, data_set_type, validation_type):
        """Validate a network, using test or validation data."""
        if data_set_type == "test":
            data_loader = self.test_data_loader
            data_set = self.data.test_data_set
            number_of_snapshots = self.data.nr_test_snapshots
            offset_snapshots = self.data.nr_validation_snapshots + \
                               self.data.nr_training_snapshots

        elif data_set_type == "validation":
            data_loader = self.validation_data_loader
            data_set = self.data.validation_data_set
            number_of_snapshots = self.data.nr_validation_snapshots
            offset_snapshots = self.data.nr_training_snapshots

        else:
            raise Exception("Please select test or validation"
                            "when using this function.")
        network.eval()
        if validation_type == "ldos":
            validation_loss = []
            with torch.no_grad():
                for x, y in data_loader:
                    x = x.to(self.parameters._configuration["device"])
                    y = y.to(self.parameters._configuration["device"])
                    prediction = network(x)
                    validation_loss.append(network.calculate_loss(prediction, y)
                                           .item())

            return np.mean(validation_loss)
        elif validation_type == "band_energy":
            # Get optimal batch size and number of batches per snapshots.
            optimal_batch_size = self. \
                _correct_batch_size_for_testing(self.data.grid_size,
                                                self.parameters.
                                                mini_batch_size)
            number_of_batches_per_snapshot = int(self.data.grid_size /
                                                 optimal_batch_size)
            errors = []
            for snapshot_number in range(offset_snapshots,
                                         number_of_snapshots+offset_snapshots):
                actual_outputs, \
                predicted_outputs = self.\
                    _forward_entire_snapshot(snapshot_number-offset_snapshots,
                                             data_set,
                                             number_of_batches_per_snapshot,
                                             optimal_batch_size)
                calculator = self.data.target_calculator

                # This works because the list is always guaranteed to be
                # ordered.
                calculator.\
                    read_additional_calculation_data("qe.out",
                                                     self.data.get_snapshot_calculation_output(snapshot_number))
                fe_actual = calculator.\
                    get_self_consistent_fermi_energy(actual_outputs)
                be_actual = calculator.\
                    get_band_energy(actual_outputs, fermi_energy=fe_actual)

                try:
                    fe_predicted = calculator.\
                        get_self_consistent_fermi_energy(predicted_outputs)
                    be_predicted = calculator.\
                        get_band_energy(predicted_outputs,
                                        fermi_energy=fe_predicted)
                except ValueError:
                    # If the training went badly, it might be that the above
                    # code results in an error, due to the LDOS being so wrong
                    # that the estimation of the self consistent Fermi energy
                    # fails.
                    be_predicted = float("inf")
                errors.append(np.abs(be_predicted-be_actual) *
                              (1000/len(calculator.atoms)))
            return np.mean(errors)
        elif validation_type == "total_energy":
            # Get optimal batch size and number of batches per snapshots.
            optimal_batch_size = self. \
                _correct_batch_size_for_testing(self.data.grid_size,
                                                self.parameters.
                                                mini_batch_size)
            number_of_batches_per_snapshot = int(self.data.grid_size /
                                                 optimal_batch_size)
            errors = []
            for snapshot_number in range(offset_snapshots,
                                         number_of_snapshots+offset_snapshots):
                actual_outputs, predicted_outputs = self.\
                    _forward_entire_snapshot(snapshot_number-offset_snapshots,
                                             data_set,
                                             number_of_batches_per_snapshot,
                                             optimal_batch_size)
                calculator = self.data.target_calculator

                # This works because the list is always guaranteed to be
                # ordered.
                calculator.\
                    read_additional_calculation_data("qe.out",
                                                     self.data.get_snapshot_calculation_output(snapshot_number))
                fe_actual = calculator.\
                    get_self_consistent_fermi_energy(actual_outputs)
                te_actual = calculator.\
                    get_total_energy(ldos_data=actual_outputs,
                                     fermi_energy=fe_actual)

                try:
                    fe_predicted = calculator.\
                        get_self_consistent_fermi_energy(predicted_outputs)
                    te_predicted = calculator.\
                        get_total_energy(ldos_data=actual_outputs,
                                         fermi_energy=fe_predicted)
                except ValueError:
                    # If the training went badly, it might be that the above
                    # code results in an error, due to the LDOS being so wrong
                    # that the estimation of the self consistent Fermi energy
                    # fails.
                    te_predicted = float("inf")
                errors.append(np.abs(te_predicted-te_actual) *
                              (1000/len(calculator.atoms)))
            return np.mean(errors)

        else:
            raise Exception("Selected validation method not supported.")

    def __create_training_checkpoint(self):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        network_name = self.parameters.checkpoint_name \
            + "_network.pth"
        iscaler_name = self.parameters.checkpoint_name \
            + "_iscaler.pkl"
        oscaler_name = self.parameters.checkpoint_name \
            + "_oscaler.pkl"
        if self.use_pkl_checkpoints:
            param_name = self.parameters.checkpoint_name \
                         + "_params.pkl"
            self.parameters_full.save_as_pickle(param_name)
        else:
            param_name = self.parameters.checkpoint_name \
                         + "_params.json"
            self.parameters_full.save_as_json(param_name)

        optimizer_name = self.parameters.checkpoint_name \
            + "_optimizer.pth"

        # First we save the objects we would also save for inference.
        self.data.input_data_scaler.save(iscaler_name)
        self.data.output_data_scaler.save(oscaler_name)
        self.network.save_network(network_name)

        # Next, we save all the other objects.

        if self.parameters_full.use_horovod:
            if hvd.rank() != 0:
                return
        if self.scheduler is None:
            save_dict = {
                'epoch': self.last_epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'early_stopping_counter': self.patience_counter,
                'early_stopping_last_loss': self.last_loss
            }
        else:
            save_dict = {
                'epoch': self.last_epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.scheduler.state_dict(),
                'early_stopping_counter': self.patience_counter,
                'early_stopping_last_loss': self.last_loss
            }

        torch.save(save_dict, optimizer_name,
                   _use_new_zipfile_serialization=False)

    @staticmethod
    def __average_validation(val, name):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val)
        avg_loss = hvd.allreduce(tensor, name=name, op=hvd.Average)
        return avg_loss.item()
