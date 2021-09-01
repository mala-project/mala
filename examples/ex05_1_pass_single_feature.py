"""Show how to run a sinlge feature vector out of a snapshot through a network.

First run ex05_training_with_postprocessing.py to create the
ex05_* files used below.
"""

import pathlib as pl

import mala
import numpy as np
import torch

from data_repo_path import data_repo_path


data_path = pl.Path(data_repo_path) / "Al36"
params_path = "ex05_params.pkl"
network_path = "ex05_network.pth"
input_scaler_path = "ex05_iscaler.pkl"
output_scaler_path = "ex05_oscaler.pkl"


def load_whole_snapshot():
    """Usual MALA workflow: load whole snapshot using DataHandler. Extract
    single feature vector afterwards.
    """

    # First load Parameters and network.
    parameters = mala.Parameters.load_from_file(params_path, no_snapshots=True)

    # Load a network from a file.
    network = mala.Network.load_from_file(parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)

    inference_data_handler = mala.DataHandler(
        parameters, input_data_scaler=iscaler, output_data_scaler=oscaler
    )

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    parameters.data.data_splitting_snapshots = ["te"]
    inference_data_handler.add_snapshot(
        "Al_debug_2k_nr2.in.npy",
        data_path,
        "Al_debug_2k_nr2.out.npy",
        data_path,
        output_units="1/Ry",
    )
    inference_data_handler.prepare_data(reparametrize_scaler=False)

    # Extract single feature vector from snapshot. The x,y,z part (first there
    # entries) are removed in DataHandler (see load_memmap_snapshot() for
    # details) and parameters.descriptors_contain_xyz, which is True by
    # default. So the length is 91.
    x_scaled = inference_data_handler.test_data_inputs[0]
    with torch.no_grad():
        y_pred = network(x_scaled)

    # Convert to 1/eV
    y_rescaled = oscaler.inverse_transform(y_pred)

    return dict(x_scaled=x_scaled, y_pred=y_pred, y_rescaled=y_rescaled)


def load_memmap_snapshot():
    """Load numpy memmap of snapshot and extract single feature. Avoid loading
    whole snapshot into memory.
    """
    # First load Parameters and network.
    parameters = mala.Parameters.load_from_file(params_path, no_snapshots=True)
    # Load a network from a file.
    network = mala.Network.load_from_file(parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)

    # Load snapshot with input features, use memmap.
    inputs_array = np.load(data_path / "Al_debug_2k_nr2.in.npy", mmap_mode="r")

    # Our inputs usually contain x,y,z information, so one single input vector
    # reads [x,y,z,f_0, f_1, ..., f_N] where [f_0, f_1, ..., f_N] is the actual
    # SNAP feature vector. Now, we select a single feature vector from the
    # snapshot.
    x = torch.from_numpy(inputs_array[0, 0, 0, 3:].astype(np.float32))

    # Convert to tensor and scale.
    #
    # For some reason, DataScaler changes the shape (N,) -> (1,N), revert that
    x_scaled = iscaler.transform(x)[0, :]
    with torch.no_grad():
        y_pred = network(x_scaled)

    # Convert to 1/eV
    y_rescaled = oscaler.inverse_transform(y_pred)

    return dict(x_scaled=x_scaled, y_pred=y_pred, y_rescaled=y_rescaled, x=x)


if __name__ == "__main__":
    a = load_whole_snapshot()
    b = load_memmap_snapshot()

    for name in ["x_scaled", "y_pred", "y_rescaled"]:
        ref = a[name]
        val = b[name]
        assert (ref == val).all()
        assert ref.shape == val.shape
        assert ref.ndim == val.ndim == 1
