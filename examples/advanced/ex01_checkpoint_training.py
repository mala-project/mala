import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
Shows how a training run can be paused and 
resumed. Delete the ex07.zip file prior to execution to see the effect
of checkpointing.
Afterwards, execute this script twice to see how MALA progresses from a 
checkpoint. As the number of total epochs cannot be divided by the number
of epochs after which a checkpoint is created without residual, this will 
lead to MALA performing the missing epochs again. 
"""


def initial_setup():

    parameters = mala.Parameters()
    parameters.data.data_splitting_type = "by_snapshot"
    parameters.data.input_rescaling_type = "feature-wise-standard"
    parameters.data.output_rescaling_type = "normal"
    parameters.network.layer_activations = ["ReLU"]
    parameters.running.max_number_epochs = 9
    parameters.running.mini_batch_size = 8
    parameters.running.learning_rate = 0.00001
    parameters.running.trainingtype = "Adam"

    # We checkpoint the training every 5 epochs and save the results
    # as "ex07".
    parameters.running.checkpoints_each_epoch = 5
    parameters.running.checkpoint_name = "ex07"

    data_handler = mala.DataHandler(parameters)
    data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                              "Be_snapshot0.out.npy", data_path, "tr")
    data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                              "Be_snapshot1.out.npy", data_path, "va")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path, "te")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    parameters.network.layer_sizes = [data_handler.input_dimension,
                                           100,
                                           data_handler.output_dimension]

    # Setup network and trainer.
    test_network = mala.Network(parameters)
    test_trainer = mala.Trainer(parameters, test_network, data_handler)

    printout("Network setup: DONE.")

    return parameters, test_network, data_handler, test_trainer


def run_example07(desired_loss_improvement_factor=1):
    if mala.Trainer.run_exists("ex07"):
        parameters, network, datahandler, trainer = \
            mala.Trainer.load_run("ex07")
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
