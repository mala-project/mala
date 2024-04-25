import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
Shows how recent developments in hyperparameter optimization techniques can be 
used (OAT / training-free NAS).

REQUIRES OAPACKAGE.
"""


def optimize_hyperparameters(hyper_optimizer):

    parameters = mala.Parameters()
    parameters.data.input_rescaling_type = "feature-wise-standard"
    parameters.data.output_rescaling_type = "normal"
    parameters.running.max_number_epochs = 10
    parameters.running.mini_batch_size = 40
    parameters.running.learning_rate = 0.00001
    parameters.running.trainingtype = "Adam"
    parameters.hyperparameters.n_trials = 8
    parameters.hyperparameters.hyper_opt_method = hyper_optimizer

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################
    data_handler = mala.DataHandler(parameters)

    # Add all the snapshots we want to use in to the list.
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
    data_handler.add_snapshot(
        "Be_snapshot2.in.npy",
        data_path,
        "Be_snapshot2.out.npy",
        data_path,
        "te",
    )
    data_handler.prepare_data()
    printout("Read data: DONE.")

    hyperoptimizer = mala.HyperOpt(parameters, data_handler)
    parameters.network.layer_sizes = [
        data_handler.input_dimension,
        100,
        100,
        data_handler.output_dimension,
    ]
    hyperoptimizer.add_hyperparameter(
        "categorical", "trainingtype", choices=["Adam", "SGD"]
    )
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
    )
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
    )
    hyperoptimizer.add_hyperparameter(
        "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
    )

    hyperoptimizer.perform_study()
    hyperoptimizer.set_optimal_parameters()


optimize_hyperparameters("oat")
optimize_hyperparameters("naswot")
