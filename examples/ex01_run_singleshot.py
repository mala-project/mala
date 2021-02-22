from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.datahandling.data_handler import DataHandler
from fesl.network.network import Network
from fesl.network.trainer import Trainer
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al256_reduced/"

"""
ex01_run_singleshot.py: Shows how a neural network can be trained on material data using this framework.
It uses preprocessed data, that is read in from *.npy files.
"""

printout("Welcome to FESL.")
printout("Running ex01_run_singleshot.py")


def run_example01(desired_loss_improvement_factor=1):
    printout("Welcome to FESL.")
    printout("Running ex01_run_singleshot.py")

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = Parameters()
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"
    test_parameters.running.use_gpu = False
    test_parameters.running.use_horovod = False
    test_parameters.running.use_compression= False

    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################

    data_handler = DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path, "Al_debug_2k_nr0.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path, "Al_debug_2k_nr1.out.npy", data_path, output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path, "Al_debug_2k_nr2.out.npy", data_path, output_units="1/Ry")

    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = Network(test_parameters)
    test_trainer = Trainer(test_parameters)
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    test_trainer.train_network(test_network, data_handler)
    printout("Training: DONE.")

    ####################
    printout("Parameters used for this experiment:")
    test_parameters.show()

    if desired_loss_improvement_factor*test_trainer.initial_test_loss < test_trainer.final_test_loss:
        return False
    else:
        return True


if __name__ == "__main__":
    if run_example01():
        printout("Successfully ran ex01_run_singleshot.")
    else:
        raise Exception("Ran ex01_run_singleshot but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")
