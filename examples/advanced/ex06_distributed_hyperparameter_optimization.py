import os

import mala

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
ex09_distributed_hyperopt.py: Shows how a hyperparameter 
optimization can be sped up using a RDB storage. Ideally this should be done
using a database server system, such as PostgreSQL or MySQL. 
For this easy example, sqlite will be used. It is highly advisory not to
to use this for actual, at-scale calculations!

This example requires SQLite, which has to be installed manually.
"""


# Remove the previous data base.
try:
    os.remove("ex06.db")
except FileNotFoundError:
    pass

parameters = mala.Parameters()
# Specify the data scaling.
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
parameters.running.max_number_epochs = 5
parameters.running.mini_batch_size = 40
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"
parameters.hyperparameters.n_trials = 10
parameters.hyperparameters.checkpoints_each_trial = -1
parameters.hyperparameters.checkpoint_name = "ex06"
parameters.hyperparameters.hyper_opt_method = "optuna"
parameters.hyperparameters.study_name = "ex06"
parameters.hyperparameters.rdb_storage = "sqlite:///ex06.db"

# Hyperparameter optimization can be further refined by using ensemble training
# at each step and by using a different metric then the validation loss
# (e.g. the band energy). It is recommended not to use the ensemble training
# method in Single-GPU use, as it naturally drastically decreases performance.
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5
parameters.hyperparameters.number_training_per_trial = 3
parameters.running.after_before_training_metric = "band_energy"

data_handler = mala.DataHandler(parameters)

data_handler.add_snapshot(
    "Be_snapshot1.in.npy",
    data_path,
    "Be_snapshot1.out.npy",
    data_path,
    "tr",
    calculation_output_file=os.path.join(data_path, "Be_snapshot1.out"),
)
data_handler.add_snapshot(
    "Be_snapshot2.in.npy",
    data_path,
    "Be_snapshot2.out.npy",
    data_path,
    "va",
    calculation_output_file=os.path.join(data_path, "Be_snapshot2.out"),
)
data_handler.prepare_data()


hyperoptimizer = mala.HyperOpt(parameters, data_handler)
hyperoptimizer.add_hyperparameter("float", "learning_rate", 0.0000001, 0.01)
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
hyperoptimizer.perform_study()
hyperoptimizer.set_optimal_parameters()
