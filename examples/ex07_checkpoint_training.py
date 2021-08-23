import mala
from mala import printout
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

"""
ex07_checkpoint_training.py: Shows how a training run can be paused and 
resumed. Delete all ex07_*.pkl and ex07_*pth prior to execution.
Afterwards, execute this script twice to see how MALA progresses from a 
checkpoint. As the number of total epochs cannot be divided by the number
of epochs after which a checkpoint is created without residual, this will 
lead to MALA performing the missing epochs again. 
"""


def initial_setup():
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.model.layer_activations = ["ReLU"]

    # Specify the training parameters.
    # We only train for an odd number of epochs here, and train for
    # the rest after the checkpoint has been loaded.
    test_parameters.running.max_number_epochs = 9
    test_parameters.running.mini_batch_size = 8
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    # We checkpoint the training every 5 epochs and save the results
    # as "ex07".
    test_parameters.running.checkpoints_each_epoch = 5
    test_parameters.running.checkpoint_name = "ex07"

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
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

    test_parameters.model.layer_sizes = [data_handler.get_input_dimension(),
                                         100,
                                         data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)

    printout("Network setup: DONE.")

    return test_parameters, test_network, data_handler, test_trainer


def run_example07(desired_loss_improvement_factor=1):
    if mala.Trainer.checkpoint_exists("ex07"):
        parameters, network, datahandler, trainer = \
            mala.Trainer.resume_checkpoint("ex07")
        printout("Starting resumed training.")
    else:
        parameters, network, datahandler, trainer = initial_setup()
        printout("Starting original training.")

    ####################
    # TRAINING
    # Train the network. After training, load from the last checkpoint
    # and train for more epochs.
    ####################
    trainer.train_network()
    printout("Training: DONE.")

    ####################
    # RESULTS.
    # Print the used parameters and check whether the loss decreased enough.
    ####################

    printout("Parameters used for this experiment:")
    parameters.show()

    if desired_loss_improvement_factor*trainer.initial_test_loss\
            < trainer.final_test_loss:
        return False
    else:
        return True


if __name__ == "__main__":
    if run_example07():
        printout("Successfully ran ex07_checkpoint_training.")
    else:
        raise Exception("Ran ex07_checkpoint_training but something was off."
                        " If you haven't changed any parameters in "
                        "the example, there might be a problem with your"
                        " installation.")
