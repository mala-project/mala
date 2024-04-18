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
    parameters.running.checkpoint_name = "ex01_checkpoint"

    data_handler = mala.DataHandler(parameters)
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
    data_handler.prepare_data()

    parameters.network.layer_sizes = [
        data_handler.input_dimension,
        100,
        data_handler.output_dimension,
    ]

    test_network = mala.Network(parameters)
    test_trainer = mala.Trainer(parameters, test_network, data_handler)

    return parameters, test_network, data_handler, test_trainer


if mala.Trainer.run_exists("ex01_checkpoint"):
    parameters, network, datahandler, trainer = mala.Trainer.load_run(
        "ex01_checkpoint"
    )
    printout("Starting resumed training.")
else:
    parameters, network, datahandler, trainer = initial_setup()
    printout("Starting original training.")

trainer.train_network()
