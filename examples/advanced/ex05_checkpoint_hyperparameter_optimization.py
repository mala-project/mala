import os

import mala

from mala.datahandling.data_repo import data_path

"""
Shows how a hyperparameter optimization run can
be paused and resumed. Delete all ex04_*.pkl and ex04_*.pth prior to execution.
Afterwards, execute this script twice to see how MALA progresses from a
checkpoint. As the number of trials cannot be divided by the number
of epochs after which a checkpoint is created without residual, this will
lead to MALA performing the missing trials again.
"""


def initial_setup():
    parameters = mala.Parameters()
    parameters.data.input_rescaling_type = "feature-wise-standard"
    parameters.data.output_rescaling_type = "normal"
    parameters.running.max_number_epochs = 10
    parameters.running.mini_batch_size = 40
    parameters.running.learning_rate = 0.00001
    parameters.running.optimizer = "Adam"
    parameters.hyperparameters.n_trials = 9
    parameters.hyperparameters.checkpoints_each_trial = 5
    parameters.hyperparameters.checkpoint_name = "ex05_checkpoint"

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

    hyperoptimizer = mala.HyperOpt(parameters, data_handler)
    hyperoptimizer.add_hyperparameter(
        "float", "learning_rate", 0.0000001, 0.01
    )
    hyperoptimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10, 100)
    hyperoptimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10, 100)
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
    )
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
    )
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
    )

    return parameters, data_handler, hyperoptimizer


if mala.HyperOptOptuna.checkpoint_exists("ex05_checkpoint"):
    parameters, datahandler, hyperoptimizer = (
        mala.HyperOptOptuna.resume_checkpoint("ex05_checkpoint")
    )
else:
    parameters, datahandler, hyperoptimizer = initial_setup()

# Perform hyperparameter optimization.
hyperoptimizer.perform_study()
