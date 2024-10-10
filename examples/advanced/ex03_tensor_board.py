import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_path

"""
Shows how a NN training by MALA can be visualized using
tensorboard. The training is a basic MALA network training.
"""


parameters = mala.Parameters()
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5
parameters.network.layer_activations = ["ReLU"]
parameters.running.max_number_epochs = 100
parameters.running.mini_batch_size = 40
parameters.running.learning_rate = 0.001
parameters.running.optimizer = "Adam"

# Turn the visualization on and select a folder to save the visualization
# files into.
parameters.running.logger = "tensorboard"
parameters.running.logging_dir = "mala_vis"
parameters.running.validation_metrics = ["ldos", "band_energy"]
parameters.running.validate_every_n_epochs = 5

data_handler = mala.DataHandler(parameters)
data_handler.add_snapshot(
    "Be_snapshot0.in.npy", data_path, "Be_snapshot0.out.npy", data_path, "tr",
    calculation_output_file=os.path.join(data_path, "Be_snapshot0.out"),
)
data_handler.add_snapshot(
    "Be_snapshot1.in.npy", data_path, "Be_snapshot1.out.npy", data_path, "va",
    calculation_output_file=os.path.join(data_path, "Be_snapshot1.out"),
)
data_handler.prepare_data()
parameters.network.layer_sizes = [
    data_handler.input_dimension,
    100,
    data_handler.output_dimension,
]

network = mala.Network(parameters)
trainer = mala.Trainer(parameters, network, data_handler)
trainer.train_network()
printout(
    'Run finished, launch tensorboard with "tensorboard --logdir '
    + trainer.full_logging_path
    + '"'
)
