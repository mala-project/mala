import fesl
from fesl import printout
from data_repo_path import get_data_repo_path
import time
import numpy as np
data_path = get_data_repo_path()+"Al256_reduced/"
test_checkpoint_name = "test"


def test_run(use_gpu):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = fesl.Parameters()
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "tr", "tr", "tr",
                                                     "tr", "tr", "va", "te"]

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

    data_handler = fesl.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    for i in range(0, 6):
        data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                  "Al_debug_2k_nr0.out.npy", data_path,
                                  output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                              "Al_debug_2k_nr1.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                              "Al_debug_2k_nr2.out.npy", data_path,
                              output_units="1/Ry")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                           100,
                                           data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = fesl.Network(test_parameters)
    test_trainer = fesl.Trainer(test_parameters, test_network, data_handler)
    starttime = time.time()
    test_trainer.train_network()

    return test_trainer.final_test_loss, time.time() - starttime


def perform_gpu_test(accuracy=1e-7, performance_improvement=1.2):
    cpu_result = test_run(False)
    gpu_result = test_run(True)

    # GPU and CPU should deliver the same result.
    if np.abs(cpu_result[0] - gpu_result[0]) > accuracy:
        printout(cpu_result[0], gpu_result[0])
        return False

    # This test is not that well suited for GPU performance
    # but we should at least see some kind of speed up.
    printout(cpu_result[1], gpu_result[1])
    if gpu_result[1] > cpu_result[1] / performance_improvement:
        printout(cpu_result[1], gpu_result[1])
        return False
    return True


if __name__ == "__main__":
    if perform_gpu_test():
        printout("GPU test sucessful.")
    else:
        raise Exception("GPU test failed.")
