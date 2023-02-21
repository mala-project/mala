import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex02_test_network.py: Train a network, then use this network 
to predict the LDOS and then analyze the results of this prediction. This 
example is structured a little bit different than other examples. It consists 
of two functions, to show that networks can be saved and reaccessed after being
trained once. By default, the training function is commented out. A saved 
network architecture and parameters is provided in this repository, so this 
example focusses more on the analysis part. Nonetheless, the training can also
be done by simply uncommenting the function call. Please not that the values 
calculated at the end will be wrong, because we operate on 0.0025 % of a full 
DFT simulation cell for this example. 
"""


# Uses a trained network to make a prediction.
# The Tester class is the testing analogon to the training class.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."
parameters, network, data_handler, tester = mala.Tester.load_run("be_model")
tester.observables_to_test = ["band_energy", "number_of_electrons"]

# Specify the correct LDOS parameters.
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

# Inference should ALWAYS be done with lazy loading activated, even if
# training was not.
parameters.data.use_lazy_loading = True

# Add snapshots that are to be tested and make sure that the
# data_splitting_snapshots list is correct.
# The clear data in front is used because some data is still attached from
# the training process.
data_handler.clear_data()
data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                          "Be_snapshot2.out.npy", data_path, "te",
                          calculation_output_file=
                          os.path.join(data_path, "Be_snapshot2.out"))
data_handler.add_snapshot("Be_snapshot3.in.npy", data_path,
                          "Be_snapshot3.out.npy", data_path, "te",
                          calculation_output_file=
                          os.path.join(data_path, "Be_snapshot3.out"))
data_handler.prepare_data(reparametrize_scaler=False)


# Get the results for the snapshots.
results = tester.test_all_snapshots()
printout(results)

