import os

import numpy as np
from mala import Parameters, DataHandler, DataScaler, Network, Tester

from data_repo_path import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")
param_path = os.path.join(data_repo_path, "workflow_test/")
params_path = param_path+"workflow_test_params.pkl"
network_path = param_path+"workflow_test_network.pth"
input_scaler_path = param_path+"workflow_test_iscaler.pkl"
output_scaler_path = param_path+"workflow_test_oscaler.pkl"
accuracy_strict = 1e-16
accuracy_coarse = 5e-7


class TestInference:
    """Tests several inference/accuracy related parts."""

    def test_unit_conversion(self):
        """Test that RAM inexpensive unit conversion works."""
        new_parameters = Parameters.load_from_file(params_path,
                                                   no_snapshots=True)
        new_parameters.data.data_splitting_snapshots = ["te"]
        new_parameters.data.use_lazy_loading = False
        new_parameters.running.mini_batch_size = 50

        new_network = Network.load_from_file(new_parameters, network_path)

        # We use a data handler object to read the data we want to investigate.
        # We need to make sure that the same scaling is used.
        iscaler = DataScaler.load_from_file(input_scaler_path)
        oscaler = DataScaler.load_from_file(output_scaler_path)
        inference_data_handler = DataHandler(new_parameters,
                                             input_data_scaler=iscaler,
                                             output_data_scaler=oscaler)
        inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy",
                                            data_path,
                                            "Al_debug_2k_nr0.out.npy",
                                            data_path,
                                            output_units="1/Ry")

        inference_data_handler.prepare_data()

        # Confirm that unit conversion does not introduce any errors.

        from_file_1 = inference_data_handler.target_calculator.\
            convert_units(np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0) + ".out.npy")),
                          in_units="1/Ry")
        from_file_2 = np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0) + ".out.npy"))\
            * inference_data_handler.target_calculator.convert_units(1, in_units="1/Ry")

        assert from_file_1.sum() == from_file_2.sum()

    def test_inference_ram(self):
        """
        Test inference accuracy (for RAM based trainings).

        What is explicitly tested is that the scaling and rescaling do not
        change the values too drastically. During inference (or at least
        in the test case) this is done.
        """
        # Without lazy loading, there is a bigger numerical error in the actual
        # outputs. This is due to the fact that we scale and rescale data on 32
        # byte float arrays; lazy loading is therefore drastically advised for
        # inference/testing purposes.
        batchsizes = [46, 99, 500, 1977]
        for batchsize in batchsizes:
            actual_ldos, from_file, predicted_ldos, raw_predicted_outputs  =\
            self.__run(use_lazy_loading=False, batchsize=batchsize)
            assert np.isclose(actual_ldos.sum(), from_file.sum(),
                              atol=accuracy_coarse)

    def test_inference_lazy_loading(self):
        """
        Test inference accuracy (for lazy loading based trainings).

        What is explicitly tested is that the scaling and rescaling do not
        change the values too drastically. During inference (or at least
        in the test case) this is done.
        """
        # Without lazy loading, there is a bigger numerical error in the actual
        # outputs. This is due to the fact that we scale and rescale data on 32
        # byte float arrays; lazy loading is therefore drastically advised for
        # inference/testing purposes.
        batchsizes = [46, 99, 500, 1977]
        for batchsize in batchsizes:
            actual_ldos, from_file, predicted_ldos, raw_predicted_outputs = \
                self.__run(use_lazy_loading=True, batchsize=batchsize)
            assert np.isclose(actual_ldos.sum(), from_file.sum(),
                              atol=accuracy_strict)

    @staticmethod
    def __run(use_lazy_loading=False, batchsize=46):
        # First we load Parameters and network.
        new_parameters = Parameters.load_from_file(params_path,
                                                   no_snapshots=True)
        new_parameters.data.data_splitting_snapshots = ["te"]
        new_parameters.data.use_lazy_loading = use_lazy_loading
        new_parameters.running.mini_batch_size = batchsize

        new_network = Network.load_from_file(new_parameters, network_path)

        # We use a data handler object to read the data we want to investigate.
        # We need to make sure that the same scaling is used.
        iscaler = DataScaler.load_from_file(input_scaler_path)
        oscaler = DataScaler.load_from_file(output_scaler_path)
        inference_data_handler = DataHandler(new_parameters,
                                             input_data_scaler=iscaler,
                                             output_data_scaler=oscaler)
        inference_data_handler.add_snapshot("Al_debug_2k_nr0.in.npy",
                                            data_path,
                                            "Al_debug_2k_nr0.out.npy",
                                            data_path,
                                            output_units="1/Ry")

        inference_data_handler.prepare_data()

        tester = Tester(new_parameters, new_network, inference_data_handler)

        # Get values with current framework.
        actual_ldos, predicted_ldos = tester.test_snapshot(0)

        # Compare actual_ldos with file directly.
        # This is the only comparison that counts.
        from_file = inference_data_handler.target_calculator.\
            convert_units(np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0)+".out.npy")),
                          in_units="1/Ry")

        # Test if prediction still works.
        raw_predicted_outputs = np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0) + ".in.npy"))
        raw_predicted_outputs = inference_data_handler.\
            raw_numpy_to_converted_scaled_tensor(raw_predicted_outputs,
                                                 "in", "None")
        raw_predicted_outputs = new_network.\
            do_prediction(raw_predicted_outputs)
        raw_predicted_outputs = inference_data_handler.output_data_scaler.\
            inverse_transform(raw_predicted_outputs, as_numpy=True)

        return actual_ldos, from_file, predicted_ldos, raw_predicted_outputs
