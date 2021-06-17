import mala
from mala import printout
import numpy as np
from data_repo_path import get_data_repo_path
import torch
data_path = get_data_repo_path()+"Al36/"
"""
WIP: This is to show how atomic forces could be calculated in MALA. 
Note: You have to run ex05 locally to use this. 
"""
params_path = "./ex05_params.pkl"
network_path = "./ex05_network.pth"
input_scaler_path = "./ex05_iscaler.pkl"
output_scaler_path = "./ex05_oscaler.pkl"


def with_lazy_loading():
    new_parameters = mala.Parameters.load_from_file(params_path,
                                                    no_snapshots=True)

    # Specify the correct LDOS parameters.
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 250
    new_parameters.targets.ldos_gridspacing_ev = 0.1
    new_parameters.targets.ldos_gridoffset_ev = -10
    new_parameters.data.use_lazy_loading = True

    # Load a network from a file.
    new_network = mala.Network.load_from_file(new_parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    new_parameters.data.data_splitting_snapshots = ["te", "te"]
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                        "Al_debug_2k_nr2.out.npy", data_path,
                                        output_units="1/Ry")
    inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                        "Al_debug_2k_nr0.out.npy", data_path,
                                        output_units="1/Ry")

    inference_data_handler.prepare_data(reparametrize_scaler=False)

    tester = mala.Tester(new_parameters, new_network, inference_data_handler)

    dummy_gradient = torch.ones((inference_data_handler.grid_size,
                                 inference_data_handler.get_output_dimension()))
    actual_outputs, predicted_outputs = tester.test_snapshot(0, as_unscaled_numpy=False)
    ldos_calculator = inference_data_handler.target_calculator
    dd_dB_0 = ldos_calculator.get_atomic_forces(predicted_outputs, dummy_gradient,
                                      inference_data_handler, snapshot_number=0)
    actual_outputs, predicted_outputs = tester.test_snapshot(1, as_unscaled_numpy=False)
    ldos_calculator = inference_data_handler.target_calculator
    dd_dB_1 = ldos_calculator.get_atomic_forces(predicted_outputs, dummy_gradient,
                                      inference_data_handler, snapshot_number=1)

    return dd_dB_0, dd_dB_1


def without_lazy_loading():
    new_parameters = mala.Parameters.load_from_file(params_path,
                                                    no_snapshots=True)

    # Specify the correct LDOS parameters.
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 250
    new_parameters.targets.ldos_gridspacing_ev = 0.1
    new_parameters.targets.ldos_gridoffset_ev = -10
    new_parameters.data.use_lazy_loading = False

    # Load a network from a file.
    new_network = mala.Network.load_from_file(new_parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    new_parameters.data.data_splitting_snapshots = ["te", "te"]
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                        "Al_debug_2k_nr2.out.npy", data_path,
                                        output_units="1/Ry")
    inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                        "Al_debug_2k_nr0.out.npy", data_path,
                                        output_units="1/Ry")

    inference_data_handler.prepare_data(reparametrize_scaler=False)

    tester = mala.Tester(new_parameters, new_network, inference_data_handler)

    dummy_gradient = torch.ones((inference_data_handler.grid_size,
                                 inference_data_handler.get_output_dimension()))
    actual_outputs, predicted_outputs = tester.test_snapshot(0, as_unscaled_numpy=False)
    ldos_calculator = inference_data_handler.target_calculator
    dd_dB_0 = ldos_calculator.get_atomic_forces(predicted_outputs, dummy_gradient,
                                      inference_data_handler, snapshot_number=0)
    actual_outputs, predicted_outputs = tester.test_snapshot(1, as_unscaled_numpy=False)
    ldos_calculator = inference_data_handler.target_calculator
    dd_dB_1 = ldos_calculator.get_atomic_forces(predicted_outputs, dummy_gradient,
                                      inference_data_handler, snapshot_number=1)

    return dd_dB_0, dd_dB_1



def bypassing_mala():
    new_parameters = mala.Parameters.load_from_file(params_path,
                                                    no_snapshots=True)

    # Specify the correct LDOS parameters.
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 250
    new_parameters.targets.ldos_gridspacing_ev = 0.1
    new_parameters.targets.ldos_gridoffset_ev = -10
    new_parameters.data.use_lazy_loading = False

    # Load a network from a file.
    new_network = mala.Network.load_from_file(new_parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    new_parameters.data.data_splitting_snapshots = ["te", "te"]
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                        "Al_debug_2k_nr2.out.npy", data_path,
                                        output_units="1/Ry")
    inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                        "Al_debug_2k_nr0.out.npy", data_path,
                                        output_units="1/Ry")

    inference_data_handler.prepare_data(reparametrize_scaler=False)

    # The Tester class is the testing analogon to the training class.
    tester = mala.Tester(new_parameters, new_network, inference_data_handler)

    dummy_gradient = torch.ones((inference_data_handler.grid_size,
                                 inference_data_handler.get_output_dimension()))
    actual_outputs, predicted_outputs = tester.test_snapshot(0, as_unscaled_numpy=False)
    predicted_outputs.backward(gradient=dummy_gradient)
    inputs_test = inference_data_handler.test_data_inputs
    dd_dB_0 = (inputs_test.grad)[inference_data_handler.grid_size*0:
                               inference_data_handler.grid_size*(0+1)]
    actual_outputs, predicted_outputs = tester.test_snapshot(1, as_unscaled_numpy=False)
    predicted_outputs.backward(gradient=dummy_gradient)
    dd_dB_1 = (inputs_test.grad)[inference_data_handler.grid_size*1:
                               inference_data_handler.grid_size*(1+1)]

    return dd_dB_0, dd_dB_1

ram_snapshot0, ram_snapshot_1 = without_lazy_loading()
lazy_loading_snapshot0, lazy_loading_snapshot1 = without_lazy_loading()
direct0, direct1 = without_lazy_loading()

# These should all be the same.

print(ram_snapshot0, ram_snapshot_1)
print(lazy_loading_snapshot0, lazy_loading_snapshot1)
print(direct0, direct1)
