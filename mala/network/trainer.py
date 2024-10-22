"""Trainer class for training a network."""

import os
import time
from datetime import datetime
from packaging import version

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mala.common.parameters import printout
from mala.common.parallelizer import get_local_rank
from mala.datahandling.fast_tensor_dataset import FastTensorDataset
from mala.network.runner import Runner
from mala.datahandling.lazy_load_dataset_single import LazyLoadDatasetSingle
from mala.datahandling.multi_lazy_load_data_loader import (
    MultiLazyLoadDataLoader,
)
from tqdm.auto import trange, tqdm


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

        if self.parameters_full.use_ddp:
            printout("DDP activated, wrapping model in DDP.", min_verbosity=1)
            # JOSHR: using streams here to maintain compatibility with
            # graph capture
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                self.network = DDP(self.network)
            torch.cuda.current_stream().wait_stream(s)

        self.final_validation_loss = float("inf")
        self.initial_validation_loss = float("inf")
        self.optimizer = None
        self.scheduler = None
        self.patience_counter = 0
        self.last_epoch = 0
        self.last_loss = None
        self.training_data_loaders = []
        self.validation_data_loaders = []

        # Samplers for the ddp case.
        self.train_sampler = None
        self.validation_sampler = None

        self.__prepare_to_train(optimizer_dict)

        self.logger = None
        self.full_logging_path = None
        if self.parameters.logger is not None:
            os.makedirs(self.parameters.logging_dir, exist_ok=True)
            if self.parameters.logging_dir_append_date:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if len(self.parameters.run_name) > 0:
                    name = self.parameters.run_name + "_" + date_time
                else:
                    name = date_time
                self.full_logging_path = os.path.join(
                    self.parameters.logging_dir, name
                )
                os.makedirs(self.full_logging_path, exist_ok=True)
            else:
                self.full_logging_path = self.parameters.logging_dir

            # Set the path to log files
            if self.parameters.logger == "wandb":
                import wandb

                self.logger = wandb
            elif self.parameters.logger == "tensorboard":
                self.logger = SummaryWriter(self.full_logging_path)
            else:
                raise Exception(
                    f"Unsupported logger {self.parameters.logger}."
                )
            printout(
                "Writing logging output to",
                self.full_logging_path,
                min_verbosity=1,
            )

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
            return os.path.isfile(run_name + ".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params." + params_format
            optimizer_name = run_name + ".optimizer.pth"
            return all(
                map(
                    os.path.isfile,
                    [
                        iscaler_name,
                        oscaler_name,
                        param_name,
                        network_name,
                        optimizer_name,
                    ],
                )
            )

    @classmethod
    def load_run(
        cls,
        run_name,
        path="./",
        zip_run=True,
        params_format="json",
        load_runner=True,
        prepare_data=True,
    ):
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
        return super(Trainer, cls).load_run(
            run_name,
            path=path,
            zip_run=zip_run,
            params_format=params_format,
            load_runner=load_runner,
            prepare_data=prepare_data,
            load_with_gpu=None,
            load_with_mpi=None,
            load_with_ddp=None,
        )

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
        if params.use_ddp:
            map_location = {"cuda:%d" % 0: "cuda:%d" % get_local_rank()}
            checkpoint = torch.load(file, map_location=map_location)
        else:
            checkpoint = torch.load(file)

        # Now, create the Trainer class with it.
        loaded_trainer = Trainer(
            params, network, data, optimizer_dict=checkpoint
        )
        return loaded_trainer

    def train_network(self):
        """Train a network using data given by a DataHandler."""
        ############################
        # CALCULATE INITIAL METRICS
        ############################

        vloss = float("inf")

        # Save losses for later use.
        self.initial_validation_loss = vloss

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

        total_batch_id = 0

        for epoch in range(self.last_epoch, self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            self.network.train()

            training_loss_sum_logging = 0.0

            # Process each mini batch and save the training loss.
            training_loss_sum = torch.zeros(
                1, device=self.parameters._configuration["device"]
            )

            # train sampler
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)

            # shuffle dataset if necessary
            if isinstance(self.data.training_data_sets[0], FastTensorDataset):
                self.data.training_data_sets[0].shuffle()

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
                tsample = time.time()
                t0 = time.time()
                batchid = 0
                for loader in self.training_data_loaders:
                    t = time.time()
                    for inputs, outputs in tqdm(
                        loader,
                        desc="training",
                        disable=self.parameters_full.verbosity < 2,
                        total=len(loader),
                    ):
                        dt = time.time() - t
                        printout(f"load time: {dt}", min_verbosity=3)

                        if self.parameters.profiler_range is not None:
                            if batchid == self.parameters.profiler_range[0]:
                                torch.cuda.profiler.start()
                            if batchid == self.parameters.profiler_range[1]:
                                torch.cuda.profiler.stop()

                        torch.cuda.nvtx.range_push(f"step {batchid}")

                        torch.cuda.nvtx.range_push("data copy in")
                        t = time.time()
                        inputs = inputs.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )
                        outputs = outputs.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )
                        dt = time.time() - t
                        printout(f"data copy in time: {dt}", min_verbosity=3)
                        # data copy in
                        torch.cuda.nvtx.range_pop()

                        loss = self.__process_mini_batch(
                            self.network, inputs, outputs
                        )
                        # step
                        torch.cuda.nvtx.range_pop()
                        training_loss_sum += loss
                        training_loss_sum_logging += loss.item()

                        if (
                            batchid != 0
                            and (batchid + 1)
                            % self.parameters.training_log_interval
                            == 0
                        ):
                            torch.cuda.synchronize(
                                self.parameters._configuration["device"]
                            )
                            sample_time = time.time() - tsample
                            avg_sample_time = (
                                sample_time
                                / self.parameters.training_log_interval
                            )
                            avg_sample_tput = (
                                self.parameters.training_log_interval
                                * inputs.shape[0]
                                / sample_time
                            )
                            printout(
                                f"batch {batchid + 1}, "  # /{total_samples}, "
                                f"train avg time: {avg_sample_time} "
                                f"train avg throughput: {avg_sample_tput}",
                                min_verbosity=3,
                            )
                            tsample = time.time()

                            # summary_writer tensor board
                            if self.parameters.logger == "tensorboard":
                                training_loss_mean = (
                                    training_loss_sum_logging
                                    / self.parameters.training_log_interval
                                )
                                self.logger.add_scalars(
                                    "ldos",
                                    {"during_training": training_loss_mean},
                                    total_batch_id,
                                )
                                self.logger.close()
                                training_loss_sum_logging = 0.0
                            if self.parameters.logger == "wandb":
                                training_loss_mean = (
                                    training_loss_sum_logging
                                    / self.parameters.training_log_interval
                                )
                                self.logger.log(
                                    {
                                        "ldos_during_training": training_loss_mean
                                    },
                                    step=total_batch_id,
                                )
                                training_loss_sum_logging = 0.0

                        batchid += 1
                        total_batch_id += 1
                        t = time.time()
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
                t1 = time.time()
                printout(f"training time: {t1 - t0}", min_verbosity=2)

                # Calculate the validation loss. and output it.
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
            else:
                batchid = 0
                for loader in self.training_data_loaders:
                    for inputs, outputs in loader:
                        inputs = inputs.to(
                            self.parameters._configuration["device"]
                        )
                        outputs = outputs.to(
                            self.parameters._configuration["device"]
                        )
                        training_loss_sum += self.__process_mini_batch(
                            self.network, inputs, outputs
                        )
                        batchid += 1
                        total_batch_id += 1
            
            dataset_fractions = ["validation"]
            if self.parameters.validate_on_training_data:
                dataset_fractions.append("train")
            validation_metrics = ["ldos"]
            if (epoch != 0 and
                (epoch - 1) % self.parameters.validate_every_n_epochs == 0):
                validation_metrics = self.parameters.validation_metrics
            errors = self._validate_network(
                dataset_fractions, validation_metrics
            )
            for dataset_fraction in dataset_fractions:
                for metric in errors[dataset_fraction]:
                    errors[dataset_fraction][metric] = np.mean(
                        np.abs(errors[dataset_fraction][metric])
                    )
            vloss = errors["validation"][
                self.parameters.during_training_metric
            ]
            if self.parameters_full.use_ddp:
                vloss = self.__average_validation(
                    vloss,
                    "average_loss",
                    self.parameters._configuration["device"],
                )
            if self.parameters_full.verbosity > 1:
                printout("Errors:", errors, min_verbosity=2)
            else:
                printout(
                    f"Epoch {epoch}: validation data loss: {vloss:.3e}",
                    min_verbosity=1,
                )

            if self.parameters.logger == "tensorboard":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.add_scalars(
                            metric,
                            {
                                dataset_fraction: errors[dataset_fraction][
                                    metric
                                ]
                            },
                            total_batch_id,
                        )
                self.logger.close()
            if self.parameters.logger == "wandb":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.log(
                            {
                                f"{dataset_fraction}_{metric}": errors[
                                    dataset_fraction
                                ][metric]
                            },
                            step=total_batch_id,
                        )

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                self.data.mix_datasets()
            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if (
                    self.parameters.learning_rate_scheduler
                    == "ReduceLROnPlateau"
                ):
                    self.scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (
                    1.0 - self.parameters.early_stopping_threshold
                ):
                    self.patience_counter = 0
                    vloss_old = vloss
                else:
                    self.patience_counter += 1
                    printout(
                        "Validation accuracy has not improved enough.",
                        min_verbosity=1,
                    )
                    if (
                        self.patience_counter
                        >= self.parameters.early_stopping_epochs
                    ):
                        printout(
                            "Stopping the training, validation "
                            "accuracy has not improved for",
                            self.patience_counter,
                            "epochs.",
                            min_verbosity=1,
                        )
                        self.last_epoch = epoch
                        break

            # If checkpointing is enabled, we need to checkpoint.
            if self.parameters.checkpoints_each_epoch != 0:
                checkpoint_counter += 1
                if (
                    checkpoint_counter
                    >= self.parameters.checkpoints_each_epoch
                ):
                    printout("Checkpointing training.", min_verbosity=0)
                    self.last_epoch = epoch
                    self.last_loss = vloss_old
                    self.__create_training_checkpoint()
                    checkpoint_counter = 0

            printout(
                "Time for epoch[s]:",
                time.time() - start_time,
                min_verbosity=2,
            )

        ############################
        # CALCULATE FINAL METRICS
        ############################
        if self.parameters.after_training_metric in errors["validation"]:
            self.final_validation_loss = errors["validation"][
                self.parameters.after_training_metric
            ]
        else:
            final_errors = self._validate_network(
                ["validation"], [self.parameters.after_training_metric]
            )
            vloss = np.mean(
                final_errors["validation"][
                    self.parameters.after_training_metric
                ]
            )

            if self.parameters_full.use_ddp:
                vloss = self.__average_validation(
                    vloss,
                    "average_loss",
                    self.parameters._configuration["device"],
                )
            self.final_validation_loss = vloss

        # Clean-up for pre-fetching lazy loading.
        if self.data.parameters.use_lazy_loading_prefetch:
            self.training_data_loaders.cleanup()
            self.validation_data_loaders.cleanup()

    def _validate_network(self, data_set_fractions, metrics):
        # """Validate a network, using train or validation data."""
        self.network.eval()
        errors = {}
        for data_set_type in data_set_fractions:
            if data_set_type == "train":
                data_loaders = self.training_data_loaders
                data_sets = self.data.training_data_sets
                number_of_snapshots = self.data.nr_training_snapshots
                offset_snapshots = 0

            elif data_set_type == "validation":
                data_loaders = self.validation_data_loaders
                data_sets = self.data.validation_data_sets
                number_of_snapshots = self.data.nr_validation_snapshots
                offset_snapshots = self.data.nr_training_snapshots

            elif data_set_type == "test":
                raise Exception(
                    "You should not look at test set results during training"
                )
            else:
                raise Exception(
                    f"Dataset type ({data_set_type}) not recognized."
                )

            errors[data_set_type] = {}
            for metric in metrics:
                errors[data_set_type][metric] = []

            if isinstance(data_loaders, MultiLazyLoadDataLoader):
                loader_id = 0
                for loader in data_loaders:
                    grid_size = self.data.parameters.snapshot_directories_list[
                        loader_id + offset_snapshots
                    ].grid_size

                    actual_outputs = np.zeros(
                        (grid_size, self.data.output_dimension)
                    )
                    predicted_outputs = np.zeros(
                        (grid_size, self.data.output_dimension)
                    )
                    last_start = 0

                    for x, y in loader:

                        x = x.to(self.parameters._configuration["device"])
                        length = int(x.size()[0])
                        predicted_outputs[
                            last_start : last_start + length, :
                        ] = self.data.output_data_scaler.inverse_transform(
                            self.network(x).to("cpu"), as_numpy=True
                        )
                        actual_outputs[last_start : last_start + length, :] = (
                            self.data.output_data_scaler.inverse_transform(
                                y, as_numpy=True
                            )
                        )

                        last_start += length
                    calculated_errors = self._calculate_errors(
                        actual_outputs,
                        predicted_outputs,
                        metrics,
                        loader_id + offset_snapshots,
                    )
                    for metric in metrics:
                        errors[data_set_type][metric].append(
                            calculated_errors[metric]
                        )
                    loader_id += 1
            else:
                with torch.no_grad():
                    for snapshot_number in trange(
                        offset_snapshots,
                        number_of_snapshots + offset_snapshots,
                        desc="Validation",
                        disable=self.parameters_full.verbosity < 2,
                    ):
                        # Get optimal batch size and number of batches per snapshotss
                        grid_size = (
                            self.data.parameters.snapshot_directories_list[
                                snapshot_number
                            ].grid_size
                        )

                        optimal_batch_size = (
                            self._correct_batch_size_for_testing(
                                grid_size, self.parameters.mini_batch_size
                            )
                        )
                        number_of_batches_per_snapshot = int(
                            grid_size / optimal_batch_size
                        )

                        actual_outputs, predicted_outputs = (
                            self._forward_entire_snapshot(
                                snapshot_number,
                                data_sets[0],
                                data_set_type[0:2],
                                number_of_batches_per_snapshot,
                                optimal_batch_size,
                            )
                        )
                        calculated_errors = self._calculate_errors(
                            actual_outputs,
                            predicted_outputs,
                            metrics,
                            snapshot_number,
                        )
                        for metric in metrics:
                            errors[data_set_type][metric].append(
                                calculated_errors[metric]
                            )
        return errors

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {
            "num_workers": self.parameters.num_workers,
            "pin_memory": False,
        }
        if self.parameters_full.use_gpu:
            kwargs["pin_memory"] = True

        # Read last epoch
        if optimizer_dict is not None:
            self.last_epoch = optimizer_dict["epoch"] + 1

        # Scale the learning rate according to ddp.
        if self.parameters_full.use_ddp:
            if dist.get_world_size() > 1 and self.last_epoch == 0:
                printout(
                    "Rescaling learning rate because multiple workers are"
                    " used for training.",
                    min_verbosity=1,
                )
                self.parameters.learning_rate = (
                    self.parameters.learning_rate * dist.get_world_size()
                )

        # Choose an optimizer to use.
        if self.parameters.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.l2_regularization,
            )
        elif self.parameters.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.l2_regularization,
            )
        elif self.parameters.optimizer == "FusedAdam":
            if version.parse(torch.__version__) >= version.parse("1.13.0"):
                self.optimizer = optim.Adam(
                    self.network.parameters(),
                    lr=self.parameters.learning_rate,
                    weight_decay=self.parameters.l2_regularization,
                    fused=True,
                )
            else:
                raise Exception("Optimizer requires " "at least torch 1.13.0.")
        else:
            raise Exception("Unsupported optimizer.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            self.optimizer.load_state_dict(
                optimizer_dict["optimizer_state_dict"]
            )
            self.patience_counter = optimizer_dict["early_stopping_counter"]
            self.last_loss = optimizer_dict["early_stopping_last_loss"]

        if self.parameters_full.use_ddp:
            # scaling the batch size for multiGPU per node
            # self.batch_size= self.batch_size*hvd.local_size()

            # If lazy loading is used we do not shuffle the data points on
            # their own, but rather shuffle them
            # by shuffling the files themselves and then reading file by file
            # per epoch.
            # This shuffling is done in the dataset themselves.
            do_shuffle = self.parameters.use_shuffling_for_samplers
            if self.data.parameters.use_lazy_loading:
                do_shuffle = False

            self.train_sampler = (
                torch.utils.data.distributed.DistributedSampler(
                    self.data.training_data_sets[0],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=do_shuffle,
                )
            )
            self.validation_sampler = (
                torch.utils.data.distributed.DistributedSampler(
                    self.data.validation_data_sets[0],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                )
            )

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min",
                factor=self.parameters.learning_rate_decay,
                verbose=True,
            )
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate schedule.")
        if self.scheduler is not None and optimizer_dict is not None:
            self.scheduler.load_state_dict(
                optimizer_dict["lr_scheduler_state_dict"]
            )

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        if (
            self.data.parameters.use_lazy_loading
            or self.parameters_full.use_ddp
        ):
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        if isinstance(self.data.training_data_sets[0], FastTensorDataset):
            # Not shuffling in loader.
            # I manually shuffle the data set each epoch.
            self.training_data_loaders.append(
                DataLoader(
                    self.data.training_data_sets[0],
                    batch_size=None,
                    sampler=self.train_sampler,
                    **kwargs,
                    shuffle=False,
                )
            )
        else:
            if isinstance(
                self.data.training_data_sets[0], LazyLoadDatasetSingle
            ):
                self.training_data_loaders = MultiLazyLoadDataLoader(
                    self.data.training_data_sets, **kwargs
                )
            else:
                self.training_data_loaders.append(
                    DataLoader(
                        self.data.training_data_sets[0],
                        batch_size=self.parameters.mini_batch_size,
                        sampler=self.train_sampler,
                        **kwargs,
                        shuffle=do_shuffle,
                    )
                )

        if isinstance(self.data.validation_data_sets[0], FastTensorDataset):
            self.validation_data_loaders.append(
                DataLoader(
                    self.data.validation_data_sets[0],
                    batch_size=None,
                    sampler=self.validation_sampler,
                    **kwargs,
                )
            )
        else:
            if isinstance(
                self.data.validation_data_sets[0], LazyLoadDatasetSingle
            ):
                self.validation_data_loaders = MultiLazyLoadDataLoader(
                    self.data.validation_data_sets, **kwargs
                )
            else:
                self.validation_data_loaders.append(
                    DataLoader(
                        self.data.validation_data_sets[0],
                        batch_size=self.parameters.mini_batch_size * 1,
                        sampler=self.validation_sampler,
                        **kwargs,
                    )
                )

    def __process_mini_batch(self, network, input_data, target_data):
        """Process a mini batch."""
        if self.parameters._configuration["gpu"]:
            if self.parameters.use_graphs and self.train_graph is None:
                printout("Capturing CUDA graph for training.", min_verbosity=2)
                s = torch.cuda.Stream(self.parameters._configuration["device"])
                s.wait_stream(
                    torch.cuda.current_stream(
                        self.parameters._configuration["device"]
                    )
                )
                # Warmup for graphs
                with torch.cuda.stream(s):
                    for _ in range(20):
                        self.network.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(
                            enabled=self.parameters.use_mixed_precision
                        ):
                            prediction = network(input_data)
                            if self.parameters_full.use_ddp:
                                # JOSHR: We have to use "module" here to access custom method of DDP wrapped model
                                loss = network.module.calculate_loss(
                                    prediction, target_data
                                )
                            else:
                                loss = network.calculate_loss(
                                    prediction, target_data
                                )

                        if self.gradscaler:
                            self.gradscaler.scale(loss).backward()
                        else:
                            loss.backward()
                torch.cuda.current_stream(
                    self.parameters._configuration["device"]
                ).wait_stream(s)

                # Create static entry point tensors to graph
                self.static_input_data = torch.empty_like(input_data)
                self.static_target_data = torch.empty_like(target_data)

                # Capture graph
                self.train_graph = torch.cuda.CUDAGraph()
                network.zero_grad(set_to_none=True)
                with torch.cuda.graph(self.train_graph):
                    with torch.cuda.amp.autocast(
                        enabled=self.parameters.use_mixed_precision
                    ):
                        self.static_prediction = network(
                            self.static_input_data
                        )

                        if self.parameters_full.use_ddp:
                            self.static_loss = network.module.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )
                        else:
                            self.static_loss = network.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )

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

                with torch.cuda.amp.autocast(
                    enabled=self.parameters.use_mixed_precision
                ):
                    torch.cuda.nvtx.range_push("forward")
                    t = time.time()
                    prediction = network(input_data)
                    dt = time.time() - t
                    printout(f"forward time: {dt}", min_verbosity=3)
                    # forward
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("loss")
                    if self.parameters_full.use_ddp:
                        loss = network.module.calculate_loss(
                            prediction, target_data
                        )
                    else:
                        loss = network.calculate_loss(prediction, target_data)
                    dt = time.time() - t
                    printout(f"loss time: {dt}", min_verbosity=3)
                    # loss
                    torch.cuda.nvtx.range_pop()

                if self.gradscaler:
                    self.gradscaler.scale(loss).backward()
                else:
                    loss.backward()

            t = time.time()
            torch.cuda.nvtx.range_push("optimizer")
            if self.gradscaler:
                self.gradscaler.step(self.optimizer)
                self.gradscaler.update()
            else:
                self.optimizer.step()
            dt = time.time() - t
            printout(f"optimizer time: {dt}", min_verbosity=3)
            torch.cuda.nvtx.range_pop()  # optimizer

            if self.train_graph:
                return self.static_loss
            else:
                return loss
        else:
            prediction = network(input_data)
            if self.parameters_full.use_ddp:
                loss = network.module.calculate_loss(prediction, target_data)
            else:
                loss = network.calculate_loss(prediction, target_data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss

    def __create_training_checkpoint(self):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        optimizer_name = self.parameters.checkpoint_name + ".optimizer.pth"

        # Next, we save all the other objects.

        if self.parameters_full.use_ddp:
            if dist.get_rank() != 0:
                return
        if self.scheduler is None:
            save_dict = {
                "epoch": self.last_epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        else:
            save_dict = {
                "epoch": self.last_epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.scheduler.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        torch.save(
            save_dict, optimizer_name, _use_new_zipfile_serialization=False
        )
        if self.parameters.run_name != "":
            self.save_run(
                self.parameters.checkpoint_name,
                save_runner=True,
                path=self.parameters.run_name,
            )
        else:
            self.save_run(self.parameters.checkpoint_name, save_runner=True)

    @staticmethod
    def __average_validation(val, name, device="cpu"):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val, device=device)
        dist.all_reduce(tensor)
        avg_loss = tensor / dist.get_world_size()
        return avg_loss.item()
