import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_path

"""
This example shows how a trained network can be tested
with additional test snapshots. Either execute ex01 before executing this one
or download the appropriate model from the provided test data repo.
"""

model_name = "Be_model"
model_path = "./" if os.path.exists("Be_model.zip") else data_path

####################
# 1. LOADING A NETWORK
# To use the tester class to test an ML-DFT model, simply load it via the
# Tester class interface. Afterwards, set the necessary parameters.
# It is recommended to enable the "lazy-loading" feature, so that
# data is loaded into memory one snapshot at a time during testing - this
# helps keep RAM requirement down. Furthermore, you have to decide which
# observables to test (usual choices are "band_energy", "total_energy")
# and whether you want the results per snapshot
# (output_format="list") or as an averaged value (output_format="mae")
####################

parameters, network, data_handler, tester = mala.Tester.load_run(
    run_name=model_name, path=model_path
)
tester.observables_to_test = ["band_energy", "density"]
tester.output_format = "list"
parameters.data.use_lazy_loading = True

####################
# 2. DATA
# Data has to be added to the MALA workflow.
# Use the "te" function to specify that this is testing data.
# When preparing the data, make sure to select "reparametrize_scalers=False",
# since data scaling was initialized during model training.
####################

# Add precomputed snapshots.
data_handler.add_snapshot(
    "Be_snapshot2.in.npy",
    data_path,
    "Be_snapshot2.out.npy",
    data_path,
    "te",
    calculation_output_file=os.path.join(data_path, "Be_snapshot2.info.json"),
)
data_handler.add_snapshot(
    "Be_snapshot3.in.npy",
    data_path,
    "Be_snapshot3.out.npy",
    data_path,
    "te",
    calculation_output_file=os.path.join(data_path, "Be_snapshot3.info.json"),
)

# Add snapshots with "raw" (=MALA formatted) JSON, computation of descriptors
# will be performed "on-the-fly".
# data_handler.add_snapshot(
#     "Be_snapshot2.info.json",
#     data_path,
#     "Be_snapshot2.out.npy",
#     data_path,
#     "te",
#     calculation_output_file=os.path.join(data_path, "Be_snapshot2.info.json"),
# )
# data_handler.add_snapshot(
#     "Be_snapshot3.info.json",
#     data_path,
#     "Be_snapshot3.out.npy",
#     data_path,
#     "te",
#     calculation_output_file=os.path.join(data_path, "Be_snapshot3.info.json"),
# )
data_handler.prepare_data(reparametrize_scaler=False)


####################
# 3. Testing the network
# The following function computes the accuracy of the model on the selected
# observables for all snapshots provided. The results are saved into a
# dictionary which you manipulate within python or save to, e.g., a .csv file.
####################
results = tester.test_all_snapshots()
printout(results)
