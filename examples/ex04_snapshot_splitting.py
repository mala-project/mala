from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.network.network import Network
from fesl.network.trainer import Trainer

"""
ex04_snapshot_splitting.py: Shows a more intiutive way of splitting up the data, i.e. snapshot splitting.
"""
def run_example04(desired_loss_improvement_factor=1):
    print("Welcome to FESL.")
    print("Running ex04_snapshot_splitting.py")

    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = Parameters()
    test_parameters.data.datatype_in = "*.npy"
    test_parameters.data.datatype_out = "*.npy"
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.training.max_number_epochs = 20
    test_parameters.training.mini_batch_size = 40
    test_parameters.training.learning_rate = 0.00001
    test_parameters.training.trainingtype = "Adam"

    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################

    data_handler = HandlerInterface(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", "./data/", "Al_debug_2k_nr1.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")

    data_handler.load_data()
    data_handler.prepare_data()
    print("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    ####################


    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = Network(test_parameters)
    test_trainer = Trainer(test_parameters)
    print("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    print("Starting training.")
    test_trainer.train_network(test_network, data_handler)
    print("Training: DONE.")

    ####################
    print("Parameters used for this experiment:")
    test_parameters.show()

    if desired_loss_improvement_factor*test_trainer.initial_test_loss < test_trainer.final_test_loss:
        return False
    else:
        return True

if __name__ == "__main__":
    if run_example04():
        print("Successfully ran ex04_snapshot_splitting.")
    else:
        raise Exception("Ran ex04_snapshot_splitting but something was off. If you haven't changed any parameters in "
                        "the example, there might be a problem with your installation.")