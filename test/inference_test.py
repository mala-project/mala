import importlib
import os

import pytest
import numpy as np
from mala import Parameters, DataHandler, DataScaler, Network, Tester, \
                 Trainer, Predictor, LDOS

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")
param_path = os.path.join(data_repo_path, "workflow_test/")
beryllium_path = os.path.join(data_repo_path, "Be2", "training_data")
pseudopotential_path = os.path.join(data_repo_path, "Be2")
params_path = param_path+"workflow_test_params.json"
network_path = param_path+"workflow_test_network.pth"
input_scaler_path = param_path+"workflow_test_iscaler.pkl"
output_scaler_path = param_path+"workflow_test_oscaler.pkl"
accuracy_strict = 1e-16
accuracy_coarse = 5e-7
accuracy_very_coarse = 1


class TestInference:
    """Tests several inference/accuracy related parts."""

    def test_unit_conversion(self):
        """Test that RAM inexpensive unit conversion works."""
        new_parameters = Parameters.load_from_file(params_path,
                                                   no_snapshots=True)
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
                                            data_path, "te",
                                            output_units="1/(Ry*Bohr^3)")

        inference_data_handler.prepare_data()

        # Confirm that unit conversion does not introduce any errors.

        from_file_1 = inference_data_handler.target_calculator.\
            convert_units(np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0) + ".out.npy")),
                          in_units="1/(Ry*Bohr^3)")
        from_file_2 = np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0) + ".out.npy"))\
            * inference_data_handler.target_calculator.convert_units(1, in_units="1/(Ry*Bohr^3)")

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

    @pytest.mark.skipif(importlib.util.find_spec("lammps") is None,
                        reason="LAMMPS is currently not part of the pipeline.")
    def test_predictions(self):
        """
        Test that Predictor class and Tester class give the same results.

        They in principle do the same, but use slightly different routines
        under the hood. To test this, a small network is trained, and
        afterwards, objects from bot classes are used to predict the
        number of electrons and band energy.
        """
        ####################
        # Set up and train a network to be used for the tests.
        ####################

        test_parameters = Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 10
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.running.inference_data_grid = [18, 18, 27]
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 10
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.bispectrum_switchflag = 0

        data_handler = DataHandler(test_parameters)
        data_handler.add_snapshot("Be_snapshot1.in.npy", beryllium_path,
                                  "Be_snapshot1.out.npy", beryllium_path, "tr",
                                  output_units="1/(eV*Bohr^3)")
        data_handler.add_snapshot("Be_snapshot2.in.npy", beryllium_path,
                                  "Be_snapshot2.out.npy", beryllium_path, "va",
                                  output_units="1/(eV*Bohr^3)")
        data_handler.prepare_data()

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension]
        test_network = Network(test_parameters)
        test_trainer = Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()

        ####################
        # Now, first use the Tester class to make a prediction.
        ####################

        test_parameters.data.use_lazy_loading = True
        inference_data_handler = DataHandler(test_parameters,
                                             input_data_scaler=data_handler.
                                             input_data_scaler,
                                             output_data_scaler=data_handler.
                                             output_data_scaler)
        inference_data_handler.clear_data()
        inference_data_handler.add_snapshot("Be_snapshot3.in.npy",
                                            beryllium_path,
                                            "Be_snapshot3.out.npy",
                                            beryllium_path, "te",
                                            output_units="1/(eV*Bohr^3)")
        inference_data_handler.prepare_data(reparametrize_scaler=False)

        tester = Tester(test_parameters, test_network, inference_data_handler)
        actual_ldos, predicted_ldos = tester.predict_targets(0)
        ldos_calculator = inference_data_handler.target_calculator
        ldos_calculator.read_additional_calculation_data(os.path.join(
                                                         beryllium_path,
                                                         "Be_snapshot3.out"),
                                                         "espresso-out")

        band_energy_tester_class = ldos_calculator.get_band_energy(predicted_ldos)
        nr_electrons_tester_class = ldos_calculator.\
            get_number_of_electrons(predicted_ldos)

        ####################
        # Now, use the predictor class to make the same prediction.
        ####################

        predictor = Predictor(test_parameters, test_network,
                              inference_data_handler)
        predicted_ldos = predictor.predict_from_qeout(os.path.join(
                                                         beryllium_path,
                                                         "Be_snapshot3.out"))

        # In order for the results to be the same, we have to use the same
        # parameters.
        inference_data_handler. \
            target_calculator.\
            read_additional_calculation_data(os.path.join(
                                             beryllium_path,
                                             "Be_snapshot3.out"),
                                             "espresso-out")

        nr_electrons_predictor_class = inference_data_handler.\
            target_calculator.get_number_of_electrons(predicted_ldos)
        band_energy_predictor_class = inference_data_handler.\
            target_calculator.get_band_energy(predicted_ldos)

        assert np.isclose(band_energy_predictor_class,
                          band_energy_tester_class,
                          atol=accuracy_strict)
        assert np.isclose(nr_electrons_predictor_class,
                          nr_electrons_tester_class,
                          atol=accuracy_strict)

    @pytest.mark.skipif(importlib.util.find_spec("total_energy") is None
                        or importlib.util.find_spec("lammps") is None,
                        reason="QE and LAMMPS are currently not part of the "
                               "pipeline.")
    def test_total_energy_predictions(self):
        """
        Test that total energy predictions are in principle correct.

        And of course that it does not affect accuarcy if the atomic density
        based N-scaling formula is used in the calculation.
        """

        ####################
        # Set up and train a network to be used for the tests.
        ####################

        test_parameters = Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.running.inference_data_grid = [18, 18, 27]
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 10
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.bispectrum_switchflag = 0
        test_parameters.targets.pseudopotential_path = pseudopotential_path

        data_handler = DataHandler(test_parameters)
        data_handler.add_snapshot("Be_snapshot1.in.npy", beryllium_path,
                                  "Be_snapshot1.out.npy", beryllium_path, "tr",
                                  output_units="1/(eV*Bohr^3)")
        data_handler.add_snapshot("Be_snapshot2.in.npy", beryllium_path,
                                  "Be_snapshot2.out.npy", beryllium_path, "va",
                                  output_units="1/(eV*Bohr^3)")
        data_handler.prepare_data()

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension]
        test_network = Network(test_parameters)
        test_trainer = Trainer(test_parameters, test_network,
                               data_handler)
        test_trainer.train_network()

        inference_data_handler = DataHandler(test_parameters,
                                             input_data_scaler=data_handler.
                                             input_data_scaler,
                                             output_data_scaler=data_handler.
                                             output_data_scaler)
        inference_data_handler.clear_data()

        predictor = Predictor(test_parameters, test_network,
                              inference_data_handler)
        predicted_ldos = predictor. \
            predict_from_qeout(os.path.join(beryllium_path,
                                            "Be_snapshot3.out"))

        ldos_calculator: LDOS
        ldos_calculator = inference_data_handler.target_calculator
        ldos_calculator. \
            read_additional_calculation_data(os.path.join(beryllium_path,
                                                          "Be_snapshot3.out"),
                                             "espresso-out")
        ldos_calculator.read_from_array(predicted_ldos)
        total_energy_traditional = ldos_calculator.total_energy
        test_parameters.descriptors.use_atomic_density_energy_formula = True
        ldos_calculator.read_from_array(predicted_ldos)
        total_energy_atomic_density = ldos_calculator.total_energy
        assert np.isclose(total_energy_traditional, total_energy_atomic_density,
                          atol=accuracy_coarse)
        assert np.isclose(total_energy_traditional,
                          ldos_calculator.total_energy_dft_calculation,
                          atol=accuracy_very_coarse)

    @staticmethod
    def __run(use_lazy_loading=False, batchsize=46):
        # First we load Parameters and network.
        new_parameters = Parameters.load_from_file(params_path,
                                                   no_snapshots=True)
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
                                            data_path, "te",
                                            output_units="1/(Ry*Bohr^3)")

        inference_data_handler.prepare_data()

        tester = Tester(new_parameters, new_network, inference_data_handler)

        # Get values with current framework.
        actual_ldos, predicted_ldos = tester.predict_targets(0)

        # Compare actual_ldos with file directly.
        # This is the only comparison that counts.
        from_file = inference_data_handler.target_calculator.\
            convert_units(np.load(os.path.join(data_path, "Al_debug_2k_nr" + str(0)+".out.npy")),
                          in_units="1/(Ry*Bohr^3)")

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
