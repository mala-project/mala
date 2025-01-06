import os

import mala
import numpy as np
import torch

from mala.datahandling.data_repo import data_path

# This test checks that all scaling options are working and are not messing
# up the data.
# The desired accuracy is pretty inaccurate, since we simply lose some due to
# the float32 converisons.
desired_accuracy = 10e-3


class TestScaling:
    def test_errors_and_accuracy(self):
        for scaling in [
            "feature-wise-standard",
            "standard",
            "None",
            "minmax",
            "feature-wise-minmax",
        ]:
            data = np.load(os.path.join(data_path, "Be_snapshot2.out.npy"))
            data = data.astype(np.float32)
            data = data.reshape(
                [np.prod(np.shape(data)[0:3]), np.shape(data)[3]]
            )
            data = torch.from_numpy(data).float()

            data2 = np.load(os.path.join(data_path, "Be_snapshot2.out.npy"))
            data2 = data2.astype(np.float32)
            data2 = data2.reshape(
                [np.prod(np.shape(data2)[0:3]), np.shape(data2)[3]]
            )
            data2 = torch.from_numpy(data2).float()

            scaler = mala.DataScaler(scaling)
            scaler.fit(data)
            transformed = data
            scaler.transform(transformed)
            transformed = scaler.inverse_transform(transformed)
            relative_error = torch.sum(np.abs((data2 - transformed) / data2))
            assert relative_error < desired_accuracy

    def test_array_referencing(self):
        # Asserts that even with the new in-place scaling, data is referenced
        # and not copied (unless that is explicitly asked)

        for scaling in [
            "feature-wise-standard",
            "standard",
            "None",
            "minmax",
            "feature-wise-minmax",
        ]:
            data = np.load(os.path.join(data_path, "Be_snapshot2.in.npy"))
            data = data.astype(np.float32)
            data = data.reshape(
                [np.prod(np.shape(data)[0:3]), np.shape(data)[3]]
            )
            data = torch.from_numpy(data).float()

            scaler = mala.DataScaler(scaling)
            scaler.fit(data)

            numpy_array = np.expand_dims(np.random.random(94), axis=0)
            test_data = torch.from_numpy(numpy_array)
            scaler.transform(test_data)
            scaler.inverse_transform(test_data)
            numpy_array *= 2
            assert np.isclose(
                np.sum(
                    test_data.detach().numpy().astype(np.float64) - numpy_array
                ),
                0.0,
                rtol=1e-16,
            )
