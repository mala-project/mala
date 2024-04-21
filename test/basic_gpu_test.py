#!/usr/bin/env python3
"""
GPU test

This is a very basic test of the GPU functionalities of MALA (i.e. pytorch,
which MALA relies on). Two things are tested:

1. Whether or not your system has GPU support.
2. Whether or not the GPU does what it is supposed to. For this, 
a training is performed. It is measured whether or not the utilization
of the GPU results in a speed up. 
"""
import os
import time

import mala
from mala import printout
import numpy as np
import pytest
import torch

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

test_checkpoint_name = "test"

# Define the accuracy used in the tests and a parameter to control
# that the GPU acceleration actually does something.
accuracy = 1e-6
performance_improvement = 1.2


class TestGPUExecution:
    """
    Test class for simple GPU execution.

    Tests whether a GPU is available and then the execution on it.
    """

    @pytest.mark.skipif(
        torch.cuda.is_available() is False, reason="No GPU detected."
    )
    def test_gpu_performance(self):
        """
        Test whether GPU training brings performance improvements.
        """
        cpu_result = self.__run(False)
        gpu_result = self.__run(True)

        # This test is not that well suited for GPU performance
        # but we should at least see some kind of speed up.
        assert np.isclose(cpu_result[0], gpu_result[0], atol=accuracy)
        assert gpu_result[1] > cpu_result[1] / performance_improvement

    @staticmethod
    def __run(use_gpu):
        """
        Train a network using either GPU or CPU.

        Parameters
        ----------
        use_gpu : bool
            If True, a GPU will be used, elsewise a CPU will be used for
            training.

        Returns
        -------
        results : tuple
            A tuple containing the the final loss [0] and the execution time
            [1].

        """
        ####################
        # PARAMETERS
        # All parameters are handled from a central parameters class that
        # contains subclasses.
        ####################

        test_parameters = mala.Parameters()
        # Currently, the splitting in training, validation and test set are
        # done on a "by snapshot" basis.
        test_parameters.data.data_splitting_type = "by_snapshot"

        # Specify the data scaling.
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"

        # Specify the used activation function.
        test_parameters.network.layer_activations = ["ReLU"]

        # Specify the training parameters.
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.manual_seed = 1002
        test_parameters.running.use_shuffling_for_samplers = False
        test_parameters.use_gpu = use_gpu

        ####################
        # DATA
        # Add and prepare snapshots for training.
        ####################

        data_handler = mala.DataHandler(test_parameters)

        # Add a snapshot we want to use in to the list.
        for i in range(0, 6):
            data_handler.add_snapshot(
                "Be_snapshot0.in.npy",
                data_path,
                "Be_snapshot0.out.npy",
                data_path,
                "tr",
            )
        data_handler.add_snapshot(
            "Be_snapshot1.in.npy",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
            "va",
        )
        data_handler.add_snapshot(
            "Be_snapshot2.in.npy",
            data_path,
            "Be_snapshot2.out.npy",
            data_path,
            "te",
        )
        data_handler.prepare_data()
        printout("Read data: DONE.", min_verbosity=0)

        ####################
        # NETWORK SETUP
        # Set up the network and trainer we want to use.
        # The layer sizes can be specified before reading data,
        # but it is safer this way.
        ####################

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )
        starttime = time.time()
        test_trainer.train_network()

        return test_trainer.final_test_loss, time.time() - starttime
