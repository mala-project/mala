import os

import mala
from mala import printout

from data_repo_path import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")

params_path = "./ex05_params.pkl"
network_path = "./ex05_network.pth"
input_scaler_path = "./ex05_iscaler.pkl"
output_scaler_path = "./ex05_oscaler.pkl"

# First load Parameters and network.
# Parameters may have to
new_parameters = mala.Parameters.load_from_file(params_path,
                                                no_snapshots=True)

# Specify the correct LDOS parameters.
new_parameters.targets.target_type = "LDOS"
new_parameters.targets.ldos_gridsize = 250
new_parameters.targets.ldos_gridspacing_ev = 0.1
new_parameters.targets.ldos_gridoffset_ev = -10
new_parameters.running.inference_data_grid = [100, 20, 1]

# Load a network from a file.
new_network = mala.Network.load_from_file(new_parameters, network_path)

# Make sure the same scaling is used for data handling.
iscaler = mala.DataScaler.load_from_file(input_scaler_path)
oscaler = mala.DataScaler.load_from_file(output_scaler_path)
inference_data_handler = mala.DataHandler(new_parameters,
                                          input_data_scaler=iscaler,
                                          output_data_scaler=oscaler)

predictor = mala.Predictor(new_parameters, new_network, inference_data_handler)
ldos = predictor.predict_from_qeout(os.path.join(data_path, "Al.pw.scf.out"))
printout("Sucess!")
