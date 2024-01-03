"""Trainer class for training a network."""
import os
import time
from datetime import datetime
from packaging import version

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

from mala.common.parameters import printout
from mala.common.parallelizer import parallel_warn
from mala.datahandling.fast_tensor_dataset import FastTensorDataset
from mala.network.network import Network
from mala.network.runner import Runner
from mala.datahandling.lazy_load_dataset_single import LazyLoadDatasetSingle
from mala.datahandling.multi_lazy_load_data_loader import \
    MultiLazyLoadDataLoader


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

    def __init__(self, params, network, data, optimizer_dict=None):
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
        self.training_data_loaders = []
        self.validation_data_loaders = []
        self.test_data_loaders = []

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

        self.gradscaler = None
        if self.parameters.use_mixed_precision:
            printout("Using mixed precision via AMP.", min_verbosity=1)
            self.gradscaler = torch.cuda.amp.GradScaler()

        self.train_graph = None
        self.validation_graph = None

    @classmethod
    def run_exists(cls, run_name, params_format="json", zip_run=True):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        run_name : string
            Name of the checkpoint.

        params_format : bool
            Save format of the parameters.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        if zip_run is True:
            return os.path.isfile(run_name+".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params."+params_format
            optimizer_name = run_name + ".optimizer.pth"
            return all(map(os.path.isfile, [iscaler_name, oscaler_name,
                                            param_name,
                                            network_name, optimizer_name]))

    @classmethod
    def load_run(cls, run_name, path="./", zip_run=True,
                 params_format="json", load_runner=True,
                 prepare_data=True):
        """
        Load a run.

        Parameters
        ----------
        run_name : str
            Name under which the run is saved.

        path : str
            Path where the run is saved.

        zip_run : bool
            If True, MALA will attempt to load from a .zip file. If False,
            then separate files will be attempted to be loaded.

        params_format : str
            Can be "json" or "pkl", depending on what was saved by the model.
            Default is "json".

        load_runner : bool
            If True, a Runner object will be created/loaded for further use.

        prepare_data : bool
            If True, the data will be loaded into memory. This is needed when
            continuing a model training.

        Return
        ------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved to file.

        loaded_network : mala.network.network.Network
            The network saved to file.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from file.

        new_trainer : Trainer
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        return super(Trainer, cls).load_run(run_name, path=path,
                                            zip_run=zip_run,
                                            params_format=params_format,
                                            load_runner=load_runner,
                                            prepare_data=prepare_data)

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        """
        Load a trainer from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the trainer should be created.
            Has to be compatible with network and data.

        file : string
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
        checkpoint = torch.load(file)

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

        if self.data.test_data_sets:
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
        if self.data.test_data_sets:
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
            training_loss_sum = torch.zeros(1, device=self.parameters._configuration["device"])

            # train sampler
            if self.parameters_full.use_horovod:
                self.train_sampler.set_epoch(epoch)

            # shuffle dataset if necessary
            if isinstance(self.data.training_data_sets[0], FastTensorDataset):
                self.data.training_data_sets[0].shuffle()

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(self.parameters._configuration["device"])
                tsample = time.time()
                t0 = time.time()
                batchid = 0
                for loader in self.training_data_loaders:
                    for (inputs, outputs) in loader:

                        if self.parameters.profiler_range is not None:
                            if batchid == self.parameters.profiler_range[0]:
                                torch.cuda.profiler.start()
                            if batchid == self.parameters.profiler_range[1]:
                                torch.cuda.profiler.stop()

                        torch.cuda.nvtx.range_push(f"step {batchid}")

                        torch.cuda.nvtx.range_push("data copy in")
                        inputs = inputs.to(self.parameters._configuration["device"],
                                           non_blocking=True)
                        outputs = outputs.to(self.parameters._configuration["device"],
                                             non_blocking=True)
                        # data copy in
                        torch.cuda.nvtx.range_pop()

                        loss = self.__process_mini_batch(self.network,
                                                         inputs,
                                                         outputs)
                        # step
                        torch.cuda.nvtx.range_pop()
                        training_loss_sum += loss

                        if batchid != 0 and (batchid + 1) % self.parameters.training_report_frequency == 0:
                            torch.cuda.synchronize(self.parameters._configuration["device"])
                            sample_time = time.time() - tsample
                            avg_sample_time = sample_time / self.parameters.training_report_frequency
                            avg_sample_tput = self.parameters.training_report_frequency * inputs.shape[0] / sample_time
                            printout(f"batch {batchid + 1}, "#/{total_samples}, "
                                     f"train avg time: {avg_sample_time} "
                                     f"train avg throughput: {avg_sample_tput}",
                                     min_verbosity=2)
                            tsample = time.time()
                        batchid += 1
                torch.cuda.synchronize(self.parameters._configuration["device"])
                t1 = time.time()
                printout(f"training time: {t1 - t0}", min_verbosity=2)

                training_loss = training_loss_sum.item() / batchid

                # Calculate the validation loss. and output it.
                torch.cuda.synchronize(self.parameters._configuration["device"])
            else:
                batchid = 0
                for loader in self.training_data_loaders:
                    for (inputs, outputs) in loader:
                        inputs = inputs.to(
                            self.parameters._configuration["device"])
                        outputs = outputs.to(
                            self.parameters._configuration["device"])
                        training_loss_sum += self.__process_mini_batch(self.network, inputs, outputs)
                        batchid += 1
                training_loss = training_loss_sum.item() / batchid

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

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(self.parameters._configuration["device"])

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                self.data.mix_datasets()
            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(self.parameters._configuration["device"])

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if self.parameters.learning_rate_scheduler ==\
                        "ReduceLROnPlateau":
                    self.scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (1.0 - self.parameters.
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
        if len(self.data.test_data_sets) > 0:
            tloss = self.__validate_network(self.network,
                                            "test",
                                            self.parameters.
                                            after_before_training_metric)
            if self.parameters_full.use_horovod:
                tloss = self.__average_validation(tloss, 'average_loss')
            printout("Final test data loss: ", tloss, min_verbosity=0)
        self.final_test_loss = tloss

        # Clean-up for pre-fetching lazy loading.
        if self.data.parameters.use_lazy_loading_prefetch:
            self.training_data_loaders.cleanup()
            self.validation_data_loaders.cleanup()
            if len(self.data.test_data_sets) > 0:
                self.test_data_loaders.cleanup()

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {'num_workers': self.parameters.num_workers,
                  'pin_memory': False}
        if self.parameters_full.use_gpu:
            kwargs['pin_memory'] = True

        # Read last epoch
        if optimizer_dict is not None: 
            self.last_epoch = optimizer_dict['epoch']+1

        # Scale the learning rate according to horovod.
        if self.parameters_full.use_horovod:
            if hvd.size() > 1 and self.last_epoch == 0:
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
        elif self.parameters.trainingtype == "FusedAdam":
            if version.parse(torch.__version__) >= version.parse("1.13.0"):
                self.optimizer = optim.Adam(self.network.parameters(),
                                           lr=self.parameters.learning_rate,
                                           weight_decay=self.parameters.
                                           weight_decay, fused=True)
            else:
                raise Exception("Training method requires "
                                "at least torch 1.13.0.")
        else:
            raise Exception("Unsupported training method.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            self.optimizer.\
                load_state_dict(optimizer_dict['optimizer_state_dict'])
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
                distributed.DistributedSampler(self.data.training_data_sets[0],
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=do_shuffle)

            self.validation_sampler = torch.utils.data.\
                distributed.DistributedSampler(self.data.validation_data_sets[0],
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=False)

            if self.data.test_data_sets:
                self.test_sampler = torch.utils.data.\
                    distributed.DistributedSampler(self.data.test_data_sets[0],
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
        if isinstance(self.data.training_data_sets[0], FastTensorDataset):
            # Not shuffling in loader.
            # I manually shuffle the data set each epoch.
            self.training_data_loaders.append(DataLoader(self.data.training_data_sets[0],
                                                         batch_size=None,
                                                         sampler=self.train_sampler,
                                                         **kwargs,
                                                         shuffle=False))
        else:
            if isinstance(self.data.training_data_sets[0], LazyLoadDatasetSingle):
                self.training_data_loaders = MultiLazyLoadDataLoader(self.data.training_data_sets, **kwargs)
            else:
                self.training_data_loaders.append(DataLoader(self.data.training_data_sets[0],
                                                             batch_size=self.parameters.
                                                             mini_batch_size,
                                                             sampler=self.train_sampler,
                                                             **kwargs,
                                                             shuffle=do_shuffle))

        if isinstance(self.data.validation_data_sets[0], FastTensorDataset):
            self.validation_data_loaders.append(DataLoader(self.data.validation_data_sets[0],
                                                           batch_size=None,
                                                           sampler=
                                                           self.validation_sampler,
                                                           **kwargs))
        else:
            if isinstance(self.data.validation_data_sets[0], LazyLoadDatasetSingle):
                self.validation_data_loaders = MultiLazyLoadDataLoader(self.data.validation_data_sets, **kwargs)
            else:
                self.validation_data_loaders.append(DataLoader(self.data.validation_data_sets[0],
                                                               batch_size=self.parameters.
                                                               mini_batch_size * 1,
                                                               sampler=
                                                               self.validation_sampler,
                                                               **kwargs))

        if self.data.test_data_sets:
            if isinstance(self.data.test_data_sets[0], LazyLoadDatasetSingle):
                self.test_data_loaders = MultiLazyLoadDataLoader(self.data.test_data_sets, **kwargs)
            else:
                self.test_data_loaders.append(DataLoader(self.data.test_data_sets[0],
                                                         batch_size=self.parameters.
                                                         mini_batch_size * 1,
                                                         sampler=self.test_sampler,
                                                         **kwargs))

    def __process_mini_batch(self, network, input_data, target_data):
        """Process a mini batch."""
        if self.parameters._configuration["gpu"]:
            if self.parameters.use_graphs and self.train_graph is None:
                printout("Capturing CUDA graph for training.", min_verbosity=2)
                s = torch.cuda.Stream(self.parameters._configuration["device"])
                s.wait_stream(torch.cuda.current_stream(self.parameters._configuration["device"]))
                # Warmup for graphs
                with torch.cuda.stream(s):
                    for _ in range(20):
                        self.network.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                            prediction = network(input_data)
                            loss = network.calculate_loss(prediction, target_data)

                        if self.gradscaler:
                            self.gradscaler.scale(loss).backward()
                        else:
                            loss.backward()
                torch.cuda.current_stream(self.parameters._configuration["device"]).wait_stream(s)

                # Create static entry point tensors to graph
                self.static_input_data = torch.empty_like(input_data)
                self.static_target_data = torch.empty_like(target_data)

                # Capture graph
                self.train_graph = torch.cuda.CUDAGraph()
                self.network.zero_grad(set_to_none=True)
                with torch.cuda.graph(self.train_graph):
                    with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                        self.static_prediction = network(self.static_input_data)

                        self.static_loss = network.calculate_loss(self.static_prediction, self.static_target_data)

                    if self.gradscaler:
                        self.gradscaler.scale(self.static_loss).backward()
                    else:
                        self.static_loss.backward()

            if self.train_graph:
                self.static_input_data.copy_(input_data)
                self.static_target_data.copy_(target_data)
                self.train_graph.replay()
            else:
                torch.cuda.nvtx.range_push("zero_grad")
                self.network.zero_grad(set_to_none=True)
                # zero_grad
                torch.cuda.nvtx.range_pop()

                with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                    torch.cuda.nvtx.range_push("forward")
                    prediction = network(input_data)
                    # forward
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("loss")
                    loss = network.calculate_loss(prediction, target_data)
                    # loss
                    torch.cuda.nvtx.range_pop()

                if self.gradscaler:
                    self.gradscaler.scale(loss).backward()
                else:
                    loss.backward()

            torch.cuda.nvtx.range_push("optimizer")
            if self.gradscaler:
                self.gradscaler.step(self.optimizer)
                self.gradscaler.update()
            else:
                self.optimizer.step()
            torch.cuda.nvtx.range_pop() # optimizer

            if self.train_graph:
                return self.static_loss
            else:
                return loss
        else:
            prediction = network(input_data)
            loss = network.calculate_loss(prediction, target_data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss

    def __validate_network(self, network, data_set_type, validation_type):
        """Validate a network, using test or validation data."""
        if data_set_type == "test":
            data_loaders = self.test_data_loaders
            data_sets = self.data.test_data_sets
            number_of_snapshots = self.data.nr_test_snapshots
            offset_snapshots = self.data.nr_validation_snapshots + \
                               self.data.nr_training_snapshots

        elif data_set_type == "validation":
            data_loaders = self.validation_data_loaders
            data_sets = self.data.validation_data_sets
            number_of_snapshots = self.data.nr_validation_snapshots
            offset_snapshots = self.data.nr_training_snapshots

        else:
            raise Exception("Please select test or validation"
                            "when using this function.")
        network.eval()
        if validation_type == "ldos":
            validation_loss_sum = torch.zeros(1, device=self.parameters.
                                              _configuration["device"])
            with torch.no_grad():
                if self.parameters._configuration["gpu"]:
                    report_freq = self.parameters.training_report_frequency
                    torch.cuda.synchronize(self.parameters._configuration["device"])
                    tsample = time.time()
                    batchid = 0
                    for loader in data_loaders:
                        for (x, y) in loader:
                            x = x.to(self.parameters._configuration["device"],
                                     non_blocking=True)
                            y = y.to(self.parameters._configuration["device"],
                                     non_blocking=True)

                            if self.parameters.use_graphs and self.validation_graph is None:
                                printout("Capturing CUDA graph for validation.", min_verbosity=2)
                                s = torch.cuda.Stream(self.parameters._configuration["device"])
                                s.wait_stream(torch.cuda.current_stream(self.parameters._configuration["device"]))
                                # Warmup for graphs
                                with torch.cuda.stream(s):
                                    for _ in range(20):
                                        with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                            prediction = network(x)
                                            loss = network.calculate_loss(prediction, y)
                                torch.cuda.current_stream(self.parameters._configuration["device"]).wait_stream(s)

                                # Create static entry point tensors to graph
                                self.static_input_validation = torch.empty_like(x)
                                self.static_target_validation = torch.empty_like(y)

                                # Capture graph
                                self.validation_graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(self.validation_graph):
                                    with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                        self.static_prediction_validation = network(self.static_input_validation)
                                        self.static_loss_validation = network.calculate_loss(self.static_prediction_validation, self.static_target_validation)

                            if self.validation_graph:
                                self.static_input_validation.copy_(x)
                                self.static_target_validation.copy_(y)
                                self.validation_graph.replay()
                                validation_loss_sum += self.static_loss_validation
                            else:
                                with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                    prediction = network(x)
                                    loss = network.calculate_loss(prediction, y)
                                    validation_loss_sum += loss
                            if batchid != 0 and (batchid + 1) % report_freq == 0:
                                torch.cuda.synchronize(self.parameters._configuration["device"])
                                sample_time = time.time() - tsample
                                avg_sample_time = sample_time / report_freq
                                avg_sample_tput = report_freq * x.shape[0] / sample_time
                                printout(f"batch {batchid + 1}, " #/{total_samples}, "
                                         f"validation avg time: {avg_sample_time} "
                                         f"validation avg throughput: {avg_sample_tput}",
                                         min_verbosity=2)
                                tsample = time.time()
                            batchid += 1
                    torch.cuda.synchronize(self.parameters._configuration["device"])
                else:
                    batchid = 0
                    for loader in data_loaders:
                        for (x, y) in loader:
                            x = x.to(self.parameters._configuration["device"])
                            y = y.to(self.parameters._configuration["device"])
                            prediction = network(x)
                            validation_loss_sum += \
                                network.calculate_loss(prediction, y).item()
                            batchid += 1

            validation_loss = validation_loss_sum.item() / batchid
            return validation_loss
        elif validation_type == "band_energy" or \
                validation_type == "total_energy":
            errors = []
            if isinstance(self.validation_data_loaders,
                          MultiLazyLoadDataLoader):
                loader_id = 0
                for loader in data_loaders:
                    grid_size = self.data.parameters. \
                        snapshot_directories_list[loader_id +
                                                  offset_snapshots].grid_size

                    actual_outputs = np.zeros(
                        (grid_size, self.data.output_dimension))
                    predicted_outputs = np.zeros(
                        (grid_size, self.data.output_dimension))
                    last_start = 0

                    for (x, y) in loader:

                        x = x.to(self.parameters._configuration["device"])
                        length = int(x.size()[0])
                        predicted_outputs[last_start:last_start + length,
                        :] = \
                            self.data.output_data_scaler. \
                                inverse_transform(self.network(x).
                                                  to('cpu'), as_numpy=True)
                        actual_outputs[last_start:last_start + length, :] = \
                            self.data.output_data_scaler. \
                                inverse_transform(y, as_numpy=True)

                        last_start += length
                    errors.append(self._calculate_energy_errors(actual_outputs,
                                                                predicted_outputs,
                                                                validation_type,
                                                                loader_id+offset_snapshots))
                    loader_id += 1

            else:
                for snapshot_number in range(offset_snapshots,
                                             number_of_snapshots+offset_snapshots):
                    # Get optimal batch size and number of batches per snapshotss
                    grid_size = self.data.parameters.\
                        snapshot_directories_list[snapshot_number].grid_size

                    optimal_batch_size = self. \
                        _correct_batch_size_for_testing(grid_size,
                                                        self.parameters.
                                                        mini_batch_size)
                    number_of_batches_per_snapshot = int(grid_size /
                                                         optimal_batch_size)

                    actual_outputs, \
                    predicted_outputs = self.\
                        _forward_entire_snapshot(snapshot_number,
                                                 data_sets[0], data_set_type[0:2],
                                                 number_of_batches_per_snapshot,
                                                 optimal_batch_size)

                    errors.append(self._calculate_energy_errors(actual_outputs,
                                                                predicted_outputs,
                                                                validation_type,
                                                                snapshot_number))
            return np.mean(errors)
        else:
            raise Exception("Selected validation method not supported.")

    def _calculate_energy_errors(self, actual_outputs, predicted_outputs,
                                 energy_type, snapshot_number):
        self.data.target_calculator.\
            read_additional_calculation_data(self.data.
                                             get_snapshot_calculation_output(snapshot_number))
        if energy_type == "band_energy":
            try:
                fe_actual = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(actual_outputs)
                be_actual = self.data.target_calculator. \
                    get_band_energy(actual_outputs, fermi_energy=fe_actual)

                fe_predicted = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(predicted_outputs)
                be_predicted = self.data.target_calculator. \
                    get_band_energy(predicted_outputs,
                                    fermi_energy=fe_predicted)
                return np.abs(be_predicted - be_actual) * \
                       (1000 / len(self.data.target_calculator.atoms))
            except ValueError:
                # If the training went badly, it might be that the above
                # code results in an error, due to the LDOS being so wrong
                # that the estimation of the self consistent Fermi energy
                # fails.
                return float("inf")
        elif energy_type == "total_energy":
            try:
                fe_actual = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(actual_outputs)
                be_actual = self.data.target_calculator. \
                    get_total_energy(ldos_data=actual_outputs,
                                     fermi_energy=fe_actual)

                fe_predicted = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(predicted_outputs)
                be_predicted = self.data.target_calculator. \
                    get_total_energy(ldos_data=predicted_outputs,
                                    fermi_energy=fe_predicted)
                return np.abs(be_predicted - be_actual) * \
                       (1000 / len(self.data.target_calculator.atoms))
            except ValueError:
                # If the training went badly, it might be that the above
                # code results in an error, due to the LDOS being so wrong
                # that the estimation of the self consistent Fermi energy
                # fails.
                return float("inf")

        else:
            raise Exception("Invalid energy type requested.")


    def __create_training_checkpoint(self):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        optimizer_name = self.parameters.checkpoint_name \
            + ".optimizer.pth"

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

        self.save_run(self.parameters.checkpoint_name, save_runner=True)

    @staticmethod
    def __average_validation(val, name):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val)
        avg_loss = hvd.allreduce(tensor, name=name, op=hvd.Average)
        return avg_loss.item()
