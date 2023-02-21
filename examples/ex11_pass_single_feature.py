import os

pj = os.path.join

import mala
import numpy as np
import torch


"""
Show how to run a single feature vector out of a snapshot through a network.
We load a pre-trained net and pass a single input feature vector through,
including input and output scaling.

First run ex05_training_with_postprocessing.py to create the
ex05_* files used below.
"""

data_repo_path = os.environ["MALA_DATA_REPO"]

data_path = pj(data_repo_path, "Be2")


def load_whole_snapshot():
    """Usual MALA workflow: load whole snapshot using DataHandler. Extract
    single feature vector afterwards.
    """

    # First load a run.
    parameters, network, data_handler = mala.Runner.load_run("be_model",
                                                             load_runner=False)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    #
    # Units: output_units="1/(Ry*Bohr^3)" means that the output (target) data (LDOS) has
    # unit 1/Ry. Rescaled network output, after oscaler.inverse_transform() is
    # applied (see below) will be in 1/eV.
    data_handler.clear_data()
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                                        "Be_snapshot2.out.npy", data_path, "te")
    data_handler.prepare_data(reparametrize_scaler=False)

    # Extract single feature vector from snapshot. The x,y,z part (first 3
    # entries) are removed in DataHandler, so the feature vector length is 91
    # instead of 94. See also load_memmap_snapshot() below for details. The
    # removal in DataHandler is controlled by
    # parameters.descriptors_contain_xyz, which is True by default.
    x_scaled = data_handler.test_data_inputs[0]
    with torch.no_grad():
        y_pred = network(x_scaled)

    # Convert to 1/eV
    y_rescaled = data_handler.output_data_scaler.inverse_transform(y_pred)

    return dict(x_scaled=x_scaled, y_pred=y_pred, y_rescaled=y_rescaled)


def load_memmap_snapshot():
    """Load numpy memmap of snapshot and extract single feature. Avoid loading
    whole snapshot into memory.
    """
    # First load a run.
    parameters, network, data_handler = mala.Runner.load_run("be_model",
                                                             load_runner=False)

    # Load snapshot with input features, use memmap.
    inputs_array = np.load(
        pj(data_path, "Be_snapshot2.in.npy"), mmap_mode="r"
    )

    # Our inputs usually contain x,y,z information, so one single input vector
    # reads [x,y,z,f_0, f_1, ..., f_N] where [f_0, f_1, ..., f_N] is the actual
    # bispectrum feature vector. Now, we select a single feature vector from
    # the snapshot.
    x = torch.from_numpy(inputs_array[0, 0, 0, 3:].astype(np.float32))

    # For some reason, DataScaler changes the shape (N,) -> (1,N), revert that
    x_scaled = x.detach().clone()
    x_scaled = x_scaled.reshape((1, 91))
    data_handler.input_data_scaler.transform(x_scaled)
    x_scaled = x_scaled.reshape(91)
    with torch.no_grad():
        y_pred = network(x_scaled)

    # Convert to 1/eV
    y_rescaled = data_handler.output_data_scaler.inverse_transform(y_pred)

    return dict(x_scaled=x_scaled, y_pred=y_pred, y_rescaled=y_rescaled, x=x)


if __name__ == "__main__":
    assert os.path.exists(data_path), f"{data_path} missing, link data repo " \
                                      f"first"
    assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."

    a = load_whole_snapshot()
    b = load_memmap_snapshot()

    for name in ["x_scaled", "y_pred", "y_rescaled"]:
        ref = a[name]
        val = b[name]
        assert (ref == val).all()
        assert ref.shape == val.shape
        assert ref.ndim == val.ndim == 1
