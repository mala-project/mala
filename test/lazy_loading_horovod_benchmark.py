from mala.common.parameters import Parameters
from mala.common.printout import printout
from mala.datahandling.data_handler import DataHandler
from mala.network.network import Network
from mala.network.trainer import Trainer
import numpy as np
import time
from data_repo_path import get_data_repo_path
data_path_Al = get_data_repo_path()+"Al36/"

# This is a benchmark comparing the usage of horovod, lazy loading and RAM
# based storage,.


def lazy_loading_horovod_benchmark(data_path="../examples/data/",
                                   accuracy=0.0005):

    ####################
    # PARAMETERS
    ####################
    test_parameters = Parameters()
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.network.layer_activations = ["LeakyReLU"]
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 500
    test_parameters.running.trainingtype = "Adam"
    test_parameters.comment = "Horovod / lazy loading benchmark."
    test_parameters.network.nn_type = "feed-forward"
    test_parameters.manual_seed = 2021

    ####################
    # DATA
    ####################
    results = []
    for hvduse in [False, True]:
        for ll in [True, False]:
            start_time = time.time()
            test_parameters.running.learning_rate = 0.00001
            test_parameters.data.use_lazy_loading = ll
            test_parameters.use_horovod = hvduse
            data_handler = DataHandler(test_parameters)
            data_handler.clear_data()
            data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                      "Al_debug_2k_nr0.out.npy", data_path,
                                      output_units="1/Ry",
                                      add_snapshot_as="tr")
            data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                      "Al_debug_2k_nr1.out.npy", data_path,
                                      output_units="1/Ry",
                                      add_snapshot_as="tr")
            data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                      "Al_debug_2k_nr2.out.npy", data_path,
                                      output_units="1/Ry",
                                      add_snapshot_as="tr")
            data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                      "Al_debug_2k_nr1.out.npy", data_path,
                                      output_units="1/Ry",
                                      add_snapshot_as="va")
            data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                      "Al_debug_2k_nr2.out.npy", data_path,
                                      output_units="1/Ry",
                                      add_snapshot_as="te")

            data_handler.prepare_data()
            test_parameters.network.layer_sizes = \
                [data_handler.get_input_dimension(), 100,
                 data_handler.get_output_dimension()]

            # Setup network and trainer.
            test_network = Network(test_parameters)
            test_trainer = Trainer(test_parameters, test_network, data_handler)

            hvdstring = "no horovod"
            if hvduse:
                hvdstring = "horovod"

            llstring = "data in RAM"
            if ll:
                llstring = "using lazy loading"

            results.append([hvdstring, llstring, test_trainer.
                           initial_test_loss, test_trainer.final_test_loss,
                            time.time() - start_time])

    diff = []
    # For 4 local processes I get:
    # Test:  no horovod ,  using lazy loading
    # Initial loss:  0.1342976689338684
    # Final loss:  0.10587086156010628
    # Time:  3.743736743927002
    # Test:  no horovod ,  data in RAM
    # Initial loss:  0.13430887088179588
    # Final loss:  0.10572846792638302
    # Time:  1.825883388519287
    # Test:  horovod ,  using lazy loading
    # Initial loss:  0.1342976726591587
    # Final loss:  0.10554153844714165
    # Time:  4.513132572174072
    # Test:  horovod ,  data in RAM
    # Initial loss:  0.13430887088179588
    # Final loss:  0.1053303349763155
    # Time:  3.2193074226379395

    for r in results:
        printout("Test: ", r[0], ", ", r[1])
        printout("Initial loss: ", r[2])
        printout("Final loss: ", r[3])
        printout("Time: ", r[4])
        diff.append(r[3]-r[2])
    diff = np.array(diff)

    # The loss improvements should be comparable.
    if np.std(diff) > accuracy:
        printout(np.std(diff))
        return False
    return True


if __name__ == "__main__":
    test1 = lazy_loading_horovod_benchmark(data_path=data_path_Al)
    printout("Benchmark of lazy loading, horovod and ? - success?:", test1)
