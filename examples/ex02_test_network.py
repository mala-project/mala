import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex05_training_with_postprocessing.py: Train a network, then use this network 
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
def trained_network_test(network, parameters, iscaler, oscaler):

    # Specify the correct LDOS parameters.
    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5

    # Inference should ALWAYS be done with lazy loading activated, even if
    # training was not.
    parameters.data.use_lazy_loading = True

    # Load a network from a file.
    inference_data_handler = mala.DataHandler(parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    inference_data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                                        "Be_snapshot2.out.npy", data_path, "te",
                                        calculation_output_file=os.path.join(data_path,"Be_snapshot2.out"))
    inference_data_handler.add_snapshot("Be_snapshot3.in.npy", data_path,
                                        "Be_snapshot3.out.npy", data_path, "te",
                                        calculation_output_file=os.path.join(data_path,"Be_snapshot3.out"))
    inference_data_handler.prepare_data(reparametrize_scaler=False)

    # The Tester class is the testing analogon to the training class.
    tester = mala.Tester(parameters, network, inference_data_handler,
                         observables_to_test=["band_energy",
                                              "number_of_electrons"])

    # Get the results for the snapshots.
    results = tester.test_all_snapshots()
    printout(results)


# Instead of training, these can be loaded.
parameters = mala.Parameters.load_from_file("be_model.params.json",
                                            no_snapshots=True)
input_scaler = mala.DataScaler.load_from_file("be_model.iscaler.pkl")
output_scaler = mala.DataScaler.load_from_file("be_model.oscaler.pkl")
network = mala.Network.load_from_file(parameters, "be_model.network.pth")

trained_network_test(network, parameters, input_scaler, output_scaler)

