import importlib
import os
import time

from mala import Parameters, printout, DataHandler, Network, Trainer
import numpy as np
import torch
import pytest

from mala.datahandling.data_repo import data_path

# This test compares the data scaling using the regular scaling procedure and
# the lazy-loading one (incremental fitting).

accuracy_coarse = 1e-3


class TestLazyLoading:
    """Tests different aspects surrounding lazy loading."""

    def test_scaling(self):
        """
        Test that the scaling works approximately the same for RAM and LL.

        The are some numerical differences that simply occur, but we can still
        check that the scaling results are mostly identical.
        """
        ####################
        # PARAMETERS
        ####################
        test_parameters = Parameters()
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.descriptors.bispectrum_twojmax = 11
        test_parameters.targets.ldos_gridsize = 10
        test_parameters.network.layer_activations = "LeakyReLU"
        test_parameters.running.max_number_epochs = 3
        test_parameters.running.mini_batch_size = 512
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.comment = "Lazy loading test."
        test_parameters.network.nn_type = "feed-forward"
        test_parameters.running.use_gpu = True
        test_parameters.data.use_lazy_loading = False

        ####################
        # DATA
        ####################

        dataset_tester = []
        results = []
        training_tester = []
        for scalingtype in [
            "standard",
            "minmax",
            "feature-wise-standard",
            "feature-wise-minmax",
        ]:
            comparison = [scalingtype]
            for ll_type in [True, False]:
                this_result = []
                if ll_type:
                    this_result.append("lazy-loading")
                else:
                    this_result.append("RAM")
                test_parameters.data.use_lazy_loading = ll_type
                test_parameters.data.input_rescaling_type = scalingtype
                test_parameters.data.output_rescaling_type = scalingtype
                data_handler = DataHandler(test_parameters)
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
                    "tr",
                )
                data_handler.add_snapshot(
                    "Be_snapshot2.in.npy",
                    data_path,
                    "Be_snapshot2.out.npy",
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
                if scalingtype == "standard":
                    # The lazy-loading STD equation (and to a smaller amount the
                    # mean equation) is having some small accurcay issue that
                    # I presume to be due to numerical constraints. To make a
                    # meaningful comparison it is wise to scale the value here.
                    this_result.append(
                        data_handler.input_data_scaler.total_mean
                        / data_handler.nr_training_data
                    )
                    this_result.append(
                        data_handler.input_data_scaler.total_std
                        / data_handler.nr_training_data
                    )
                    this_result.append(
                        data_handler.output_data_scaler.total_mean
                        / data_handler.nr_training_data
                    )
                    this_result.append(
                        data_handler.output_data_scaler.total_std
                        / data_handler.nr_training_data
                    )
                elif scalingtype == "minmax":
                    torch.manual_seed(2002)
                    this_result.append(
                        data_handler.input_data_scaler.total_max
                    )
                    this_result.append(
                        data_handler.input_data_scaler.total_min
                    )
                    this_result.append(
                        data_handler.output_data_scaler.total_max
                    )
                    this_result.append(
                        data_handler.output_data_scaler.total_min
                    )
                    dataset_tester.append(
                        (data_handler.training_data_sets[0][3998])[0].sum()
                        + (data_handler.training_data_sets[0][3999])[0].sum()
                        + (data_handler.training_data_sets[0][4000])[0].sum()
                        + (data_handler.training_data_sets[0][4001])[0].sum()
                    )
                    test_parameters.network.layer_sizes = [
                        data_handler.input_dimension,
                        100,
                        data_handler.output_dimension,
                    ]

                    # Setup network and trainer.
                    test_network = Network(test_parameters)
                    test_trainer = Trainer(
                        test_parameters, test_network, data_handler
                    )
                    test_trainer.train_network()
                    training_tester.append(test_trainer.final_validation_loss)

                elif scalingtype == "feature-wise-standard":
                    # The lazy-loading STD equation (and to a smaller amount the
                    # mean equation) is having some small accurcay issue that
                    # I presume to be due to numerical constraints. To make a
                    # meaningful comparison it is wise to scale the value here.
                    this_result.append(
                        torch.mean(data_handler.input_data_scaler.means)
                        / data_handler.parameters.snapshot_directories_list[
                            0
                        ].grid_size
                    )
                    this_result.append(
                        torch.mean(data_handler.input_data_scaler.stds)
                        / data_handler.parameters.snapshot_directories_list[
                            0
                        ].grid_size
                    )
                    this_result.append(
                        torch.mean(data_handler.output_data_scaler.means)
                        / data_handler.parameters.snapshot_directories_list[
                            0
                        ].grid_size
                    )
                    this_result.append(
                        torch.mean(data_handler.output_data_scaler.stds)
                        / data_handler.parameters.snapshot_directories_list[
                            0
                        ].grid_size
                    )
                elif scalingtype == "feature-wise-minmax":
                    this_result.append(
                        torch.mean(data_handler.input_data_scaler.maxs)
                    )
                    this_result.append(
                        torch.mean(data_handler.input_data_scaler.mins)
                    )
                    this_result.append(
                        torch.mean(data_handler.output_data_scaler.maxs)
                    )
                    this_result.append(
                        torch.mean(data_handler.output_data_scaler.mins)
                    )

                comparison.append(this_result)
            results.append(comparison)

        for entry in results:
            assert np.isclose(entry[1][1], entry[2][1], atol=accuracy_coarse)
            assert np.isclose(entry[1][2], entry[2][2], atol=accuracy_coarse)
            assert np.isclose(entry[1][3], entry[2][3], atol=accuracy_coarse)
            assert np.isclose(entry[1][4], entry[2][4], atol=accuracy_coarse)
            assert np.isclose(entry[1][1], entry[2][1], atol=accuracy_coarse)

        assert np.isclose(
            dataset_tester[0], dataset_tester[1], atol=accuracy_coarse
        )
        assert np.isclose(
            training_tester[0], training_tester[1], atol=accuracy_coarse
        )

    def test_prefetching(self):
        # Comparing the results of pre-fetch and without pre-fetch
        # is not entirely easy. There are two main differences:
        # 1. In prefetch, the data is shuffled on the in-snapshot level.
        #    This is possible due to the shared memory approach and leads to
        #    different (better!) accuracies.
        # 2. Even when shuffling would be disabled, there is another
        #    difference. Without prefetching, since each data point is loaded
        #    one at a time by the torch loader, we can move over snapshot
        #    boundaries. This is not possible for the prefetching algorithm,
        #    where the snapshot boundaries are absolutely respected.
        #    Imagine if you have 8748 data points per snapshot and 2 snapshots,
        #    and want to use a batch size of 40.
        #    Then without pre-fetching, we get 437 full batches and one
        #    residual batch with 16 data points (40*437 = 17480; 8748*2=17496)
        #    With pre-fetching, we get 218 full batches per snapshot,
        #    meaning 8720 data points, so for both snapshots one residual
        #    batch with 28 data points. The amount of uneven batches has an
        #    ever so subtle influence on the network training.
        #    One could make these behaviors consistent by writing a custom
        #    DataLoader for the regular LazyLoading case, but I don't see
        #    the merit in that. The pre-fetching works better and faster,
        #    and the differences are barely noticeable at all.

        without_prefetching = self._train_lazy_loading(False)
        with_prefetching = self._train_lazy_loading(True)
        assert np.isclose(
            with_prefetching, without_prefetching, atol=accuracy_coarse
        )
        assert with_prefetching < without_prefetching

    @staticmethod
    def _train_lazy_loading(prefetching):
        test_parameters = Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "minmax"
        test_parameters.network.layer_activations = "ReLU"
        test_parameters.manual_seed = 1234
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.verbosity = 1
        test_parameters.data.use_lazy_loading = True
        test_parameters.data.use_lazy_loading_prefetch = prefetching

        data_handler = DataHandler(test_parameters)

        # Add a snapshot we want to use in to the list.
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
            "tr",
        )
        data_handler.add_snapshot(
            "Be_snapshot2.in.npy",
            data_path,
            "Be_snapshot2.out.npy",
            data_path,
            "va",
        )
        data_handler.add_snapshot(
            "Be_snapshot3.in.npy",
            data_path,
            "Be_snapshot3.out.npy",
            data_path,
            "va",
        )
        data_handler.prepare_data()

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        # Setup network and trainer.
        test_network = Network(test_parameters)
        test_trainer = Trainer(test_parameters, test_network, data_handler)
        test_trainer.train_network()
        return test_trainer.final_validation_loss
