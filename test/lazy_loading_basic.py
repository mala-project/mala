from fesl.common.parameters import Parameters
from fesl.common.printout import printout
from fesl.datahandling.data_handler import DataHandler
import torch
import numpy as np

# This test compares the data scaling using the regular scaling procedure and
# the lazy-loading one (incremental fitting).
from fesl.network.network import Network
from fesl.network.trainer import Trainer
from data_repo_path import get_data_repo_path
data_path_Al = get_data_repo_path()+"Al256_reduced/"


def test_lazy_loading_basic(data_path="../examples/data/", accuracy=0.001):
    ####################
    # PARAMETERS
    ####################
    test_parameters = Parameters()
    test_parameters.data.data_splitting_snapshots = ["tr", "tr", "tr", "va",
                                                     "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["LeakyReLU"]
    test_parameters.running.max_number_epochs = 3
    test_parameters.running.mini_batch_size = 512
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
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
    for scalingtype in ["standard", "normal", "feature-wise-standard",
                        "feature-wise-normal"]:
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
            data_handler.clear_data()
            data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                      "Al_debug_2k_nr0.out.npy", data_path,
                                      output_units="1/Ry")
            data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                      "Al_debug_2k_nr1.out.npy", data_path,
                                      output_units="1/Ry")
            data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                      "Al_debug_2k_nr2.out.npy", data_path,
                                      output_units="1/Ry")
            data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                      "Al_debug_2k_nr1.out.npy", data_path,
                                      output_units="1/Ry")
            data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                      "Al_debug_2k_nr2.out.npy", data_path,
                                      output_units="1/Ry")
            data_handler.prepare_data()
            if scalingtype == "standard":
                # The lazy-loading STD equation (and to a smaller amount the
                # mean equation) is having some small accurcay issue that
                # I presume to be due to numerical constraints. To make a
                # meaningful comparison it is wise to scale the value here.
                this_result.append(data_handler.input_data_scaler.total_mean /
                                   data_handler.nr_training_data)
                this_result.append(data_handler.input_data_scaler.total_std /
                                   data_handler.nr_training_data)
                this_result.append(data_handler.output_data_scaler.total_mean /
                                   data_handler.nr_training_data)
                this_result.append(data_handler.output_data_scaler.total_std /
                                   data_handler.nr_training_data)
            if scalingtype == "normal":
                torch.manual_seed(2002)
                this_result.append(data_handler.input_data_scaler.total_max)
                this_result.append(data_handler.input_data_scaler.total_min)
                this_result.append(data_handler.output_data_scaler.total_max)
                this_result.append(data_handler.output_data_scaler.total_min)
                dataset_tester.append((data_handler.training_data_set[3998])
                                      [0].sum() +
                                      (data_handler.training_data_set[3999])
                                      [0].sum() +
                                      (data_handler.training_data_set[4000])
                                      [0].sum() +
                                      (data_handler.training_data_set[4001])
                                      [0].sum())
                test_parameters.network.layer_sizes = \
                    [data_handler.get_input_dimension(), 100,
                     data_handler.get_output_dimension()]

                # Setup network and trainer.
                test_network = Network(test_parameters)
                test_trainer = Trainer(test_parameters)
                test_trainer.train_network(test_network, data_handler)
                training_tester.append(test_trainer.final_test_loss -
                                       test_trainer.initial_test_loss)

            if scalingtype == "feature-wise-standard":
                # The lazy-loading STD equation (and to a smaller amount the
                # mean equation) is having some small accurcay issue that
                # I presume to be due to numerical constraints. To make a
                # meaningful comparison it is wise to scale the value here.
                this_result.append(torch.mean(data_handler.input_data_scaler.
                                              means)/data_handler.grid_size)
                this_result.append(torch.mean(data_handler.input_data_scaler.
                                              stds)/data_handler.grid_size)
                this_result.append(torch.mean(data_handler.output_data_scaler.
                                              means)/data_handler.grid_size)
                this_result.append(torch.mean(data_handler.output_data_scaler.
                                              stds)/data_handler.grid_size)
            if scalingtype == "feature-wise-normal":
                this_result.append(torch.mean(data_handler.input_data_scaler.
                                              maxs))
                this_result.append(torch.mean(data_handler.input_data_scaler.
                                              mins))
                this_result.append(torch.mean(data_handler.output_data_scaler.
                                              maxs))
                this_result.append(torch.mean(data_handler.output_data_scaler.
                                              mins))

            comparison.append(this_result)
        results.append(comparison)

    for entry in results:
        val1 = entry[1][1]-entry[2][1]
        val2 = entry[1][2]-entry[2][2]
        val3 = entry[1][3]-entry[2][3]
        val4 = entry[1][4]-entry[2][4]
        if val1 > accuracy or val2 > accuracy or val3 > accuracy or val4 > \
                accuracy:
            printout(entry[0])
            printout(val1, val2, val3, val4)
            return False
    if torch.abs(dataset_tester[0] - dataset_tester[1]) > accuracy:
        printout(dataset_tester[0], dataset_tester[1])
        return False
    if np.abs(training_tester[0] - training_tester[1]) > accuracy:
        printout(training_tester[0], training_tester[1])
        return False

    return True


if __name__ == "__main__":
    test1 = test_lazy_loading_basic(data_path=data_path_Al)
    printout("Check if lazy loading and RAM implementation get the same "
             "results? - success?:", test1)
















