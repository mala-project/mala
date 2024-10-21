import importlib
import os

import mala
import numpy as np
import pytest

from mala.datahandling.data_repo import data_path

# Control how much the loss should be better after training compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 1

# Control the accuracies for the postprocessing routines.
accuracy_electrons = 1e-11
accuracy_total_energy = 1.5
accuracy_band_energy = 1
accuracy_predictions = 1.0
accuracy_coarse = 5e-7
accuracy_very_coarse = 3
accuracy_strict = 1e-16


class TestFullWorkflow:
    """Tests an entire MALA workflow."""

    def test_network_training(self):
        """Test whether MALA can train a NN."""

        test_trainer = self.__simple_training()
        assert test_trainer.final_validation_loss < np.inf

    def test_network_training_openpmd(self):
        """Test whether MALA can train a NN."""

        test_trainer = self.__simple_training(use_openpmd_data=True)
        assert test_trainer.final_validation_loss < np.inf

    def test_network_training_fast_dataset(self):
        """Test whether MALA can train a NN."""

        test_trainer = self.__simple_training(use_fast_tensor_dataset=True)
        assert test_trainer.final_validation_loss < np.inf

    def test_preprocessing(self):
        """
        Test whether MALA can preprocess data.

        This means reading the LDOS from cube files and calculating
        bispectrum descriptors.
        The data necessary for this is currently not in the data repo!
        """

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 4
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.descriptors_contain_xyz = True
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5

        # Create a DataConverter, and add snapshots to it.
        data_converter = mala.DataConverter(test_parameters)
        data_converter.add_snapshot(
            descriptor_input_type="espresso-out",
            descriptor_input_path=os.path.join(data_path, "Be_snapshot0.out"),
            target_input_type=".cube",
            target_input_path=os.path.join(
                data_path, "cubes", "tmp.pp*Be_ldos.cube"
            ),
            target_units="1/(Ry*Bohr^3)",
        )
        data_converter.convert_snapshots(
            complete_save_path="./", naming_scheme="Be_snapshot*"
        )

        # Compare against
        input_data = np.load("Be_snapshot0.in.npy")
        input_data_shape = np.shape(input_data)
        assert (
            input_data_shape[0] == 18
            and input_data_shape[1] == 18
            and input_data_shape[2] == 27
            and input_data_shape[3] == 17
        )

        output_data = np.load("Be_snapshot0.out.npy")
        output_data_shape = np.shape(output_data)
        assert (
            output_data_shape[0] == 18
            and output_data_shape[1] == 18
            and output_data_shape[2] == 27
            and output_data_shape[3] == 11
        )

    def test_preprocessing_openpmd(self):
        """
        Test whether MALA can preprocess data.

        This means reading the LDOS from cube files and calculating
        bispectrum descriptors.
        The data necessary for this is currently not in the data repo!
        """

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 4
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.descriptors_contain_xyz = True
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5

        # Create a DataConverter, and add snapshots to it.
        data_converter = mala.DataConverter(test_parameters)
        data_converter.add_snapshot(
            descriptor_input_type="espresso-out",
            descriptor_input_path=os.path.join(data_path, "Be_snapshot0.out"),
            target_input_type=".cube",
            target_input_path=os.path.join(
                data_path, "cubes", "tmp.pp*Be_ldos.cube"
            ),
            target_units="1/(Ry*Bohr^3)",
        )
        data_converter.convert_snapshots(
            complete_save_path="./", naming_scheme="Be_snapshot*.h5"
        )

        # Compare against
        input_data = (
            data_converter.descriptor_calculator.read_from_openpmd_file(
                "Be_snapshot0.in.h5"
            )
        )
        input_data_shape = np.shape(input_data)
        assert (
            input_data_shape[0] == 18
            and input_data_shape[1] == 18
            and input_data_shape[2] == 27
            and input_data_shape[3] == 14
        )

        output_data = data_converter.target_calculator.read_from_openpmd_file(
            "Be_snapshot0.out.h5"
        )
        output_data_shape = np.shape(output_data)
        assert (
            output_data_shape[0] == 18
            and output_data_shape[1] == 18
            and output_data_shape[2] == 27
            and output_data_shape[3] == 11
        )

    def test_postprocessing_from_dos(self):
        """
        Test whether MALA can postprocess data (DOS).

        This means calculating band energy and number of electrons from
        LDOS. Total energy is tested further below.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "DOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5

        # Create a target calculator to perform postprocessing.
        dos = mala.Target(test_parameters)
        dos.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot0.out"), "espresso-out"
        )
        dos_data = np.load(os.path.join(data_path, "Be_snapshot0.dos.npy"))

        # Calculate energies
        self_consistent_fermi_energy = dos.get_self_consistent_fermi_energy(
            dos_data
        )
        band_energy = dos.get_band_energy(dos_data)

        assert np.isclose(
            band_energy,
            dos.band_energy_dft_calculation,
            atol=accuracy_band_energy,
        )

    def test_postprocessing(self):
        """
        Test whether MALA can postprocess data (from LDOS)

        This means calculating band energy and number of electrons from
        LDOS. Total energy is tested further below.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5

        # Create a target calculator to perform postprocessing.
        ldos = mala.Target(test_parameters)
        ldos.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot0.out"), "espresso-out"
        )
        ldos_data = np.load(os.path.join(data_path, "Be_snapshot0.out.npy"))

        # Calculate energies
        self_consistent_fermi_energy = ldos.get_self_consistent_fermi_energy(
            ldos_data
        )
        band_energy = ldos.get_band_energy(
            ldos_data, fermi_energy=self_consistent_fermi_energy
        )

        assert np.isclose(
            band_energy,
            ldos.band_energy_dft_calculation,
            atol=accuracy_band_energy,
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("total_energy") is None,
        reason="QE is currently not part of the pipeline.",
    )
    def test_total_energy_from_dos_density(self):
        """
        Test whether MALA can calculate the total energy using the DOS+Density.

        This means calculating energies from the DOS and density.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.targets.pseudopotential_path = data_path
        # Create a target calculator to perform postprocessing.
        ldos = mala.Target(test_parameters)
        ldos.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot0.out"), "espresso-out"
        )
        dos_data = np.load(os.path.join(data_path, "Be_snapshot0.dos.npy"))
        dens_data = np.load(os.path.join(data_path, "Be_snapshot0.dens.npy"))

        dos = mala.DOS.from_ldos_calculator(ldos)

        # Calculate energies
        self_consistent_fermi_energy = dos.get_self_consistent_fermi_energy(
            dos_data
        )

        total_energy = ldos.get_total_energy(
            dos_data=dos_data,
            density_data=dens_data,
            fermi_energy=self_consistent_fermi_energy,
        )
        assert np.isclose(
            total_energy,
            ldos.total_energy_dft_calculation,
            atol=accuracy_total_energy,
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("total_energy") is None,
        reason="QE is currently not part of the pipeline.",
    )
    def test_total_energy_from_ldos(self):
        """
        Test whether MALA can calculate the total energy using the LDOS.

        This means calculating energies from the LDOS.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.targets.pseudopotential_path = data_path

        # Create a target calculator to perform postprocessing.
        ldos = mala.Target(test_parameters)
        ldos.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot0.out"), "espresso-out"
        )
        ldos_data = np.load(os.path.join(data_path, "Be_snapshot0.out.npy"))

        # Calculate energies
        self_consistent_fermi_energy = ldos.get_self_consistent_fermi_energy(
            ldos_data
        )
        total_energy = ldos.get_total_energy(
            ldos_data, fermi_energy=self_consistent_fermi_energy
        )
        assert np.isclose(
            total_energy,
            ldos.total_energy_dft_calculation,
            atol=accuracy_total_energy,
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("total_energy") is None,
        reason="QE is currently not part of the pipeline.",
    )
    def test_total_energy_from_ldos_openpmd(self):
        """
        Test whether MALA can calculate the total energy using the LDOS.

        This means calculating energies from the LDOS.
        """
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.targets.pseudopotential_path = data_path

        # Create a target calculator to perform postprocessing.
        ldos = mala.Target(test_parameters)
        ldos_data = ldos.read_from_openpmd_file(
            os.path.join(data_path, "Be_snapshot0.out.h5")
        )
        ldos.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot0.out"), "espresso-out"
        )

        # Calculate energies
        self_consistent_fermi_energy = ldos.get_self_consistent_fermi_energy(
            ldos_data
        )
        total_energy = ldos.get_total_energy(
            ldos_data, fermi_energy=self_consistent_fermi_energy
        )
        assert np.isclose(
            total_energy,
            ldos.total_energy_dft_calculation,
            atol=accuracy_total_energy,
        )

    def test_training_with_postprocessing_data_repo(self):
        """
        Test if a trained network can be loaded for postprocessing.

        For this test, a network from the data repo will be loaded.
        If this does not work, it's most likely because someting in the MALA
        parameters changed.
        """
        # Load parameters, network and data scalers.
        parameters, network, data_handler, tester = mala.Tester.load_run(
            "Be_model", path=data_path
        )

        parameters.targets.target_type = "LDOS"
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5
        parameters.targets.ldos_gridoffset_ev = -5
        parameters.data.use_lazy_loading = True

        data_handler.add_snapshot(
            "Be_snapshot2.in.npy",
            data_path,
            "Be_snapshot2.out.npy",
            data_path,
            "te",
            calculation_output_file=os.path.join(
                data_path, "Be_snapshot2.out"
            ),
        )
        data_handler.prepare_data(reparametrize_scaler=False)

        # Instantiate and use a Tester object.
        tester.observables_to_test = ["band_energy"]
        errors = tester.test_snapshot(0)

        # Check whether the prediction is accurate enough.
        assert np.isclose(
            errors["band_energy"], 0, atol=accuracy_predictions * 1000
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("lammps") is None,
        reason="LAMMPS is currently not part of the pipeline.",
    )
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

        parameters, network, data_handler, tester = mala.Tester.load_run(
            "Be_model", path=data_path
        )
        parameters.targets.target_type = "LDOS"
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5
        parameters.targets.ldos_gridoffset_ev = -5
        parameters.running.inference_data_grid = [18, 18, 27]
        parameters.descriptors.descriptor_type = "Bispectrum"
        parameters.descriptors.bispectrum_twojmax = 10
        parameters.descriptors.bispectrum_cutoff = 4.67637
        parameters.data.use_lazy_loading = True

        data_handler.add_snapshot(
            "Be_snapshot3.in.npy",
            data_path,
            "Be_snapshot3.out.npy",
            data_path,
            "te",
        )
        data_handler.prepare_data(reparametrize_scaler=False)

        actual_ldos, predicted_ldos = tester.predict_targets(0)
        ldos_calculator = data_handler.target_calculator
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot3.out"), "espresso-out"
        )

        band_energy_tester_class = ldos_calculator.get_band_energy(
            predicted_ldos
        )

        ####################
        # Now, use the predictor class to make the same prediction.
        ####################

        predictor = mala.Predictor(parameters, network, data_handler)
        predicted_ldos = predictor.predict_from_qeout(
            os.path.join(data_path, "Be_snapshot3.out")
        )

        # In order for the results to be the same, we have to use the same
        # parameters.
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot3.out"), "espresso-out"
        )
        band_energy_predictor_class = (
            data_handler.target_calculator.get_band_energy(predicted_ldos)
        )

        assert np.isclose(
            band_energy_predictor_class,
            band_energy_tester_class,
            atol=accuracy_strict,
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("total_energy") is None
        or importlib.util.find_spec("lammps") is None,
        reason="QE and LAMMPS are currently not part of the " "pipeline.",
    )
    def test_total_energy_predictions(self):
        """
        Test that total energy predictions are in principle correct.

        And of course that it does not affect accuarcy if the atomic density
        based N-scaling formula is used in the calculation.
        """

        ####################
        # Set up and train a network to be used for the tests.
        ####################

        parameters, network, data_handler, predictor = mala.Predictor.load_run(
            "Be_model", path=data_path
        )
        parameters.targets.target_type = "LDOS"
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5
        parameters.targets.ldos_gridoffset_ev = -5
        parameters.running.inference_data_grid = [18, 18, 27]
        parameters.descriptors.descriptor_type = "Bispectrum"
        parameters.descriptors.bispectrum_twojmax = 10
        parameters.descriptors.bispectrum_cutoff = 4.67637
        parameters.targets.pseudopotential_path = data_path

        predicted_ldos = predictor.predict_from_qeout(
            os.path.join(data_path, "Be_snapshot3.out")
        )

        ldos_calculator: mala.LDOS
        ldos_calculator = data_handler.target_calculator
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot3.out"), "espresso-out"
        )
        ldos_calculator.read_from_array(predicted_ldos)
        total_energy_traditional = ldos_calculator.total_energy
        parameters.descriptors.use_atomic_density_energy_formula = True
        ldos_calculator.read_from_array(predicted_ldos)
        total_energy_atomic_density = ldos_calculator.total_energy
        assert np.isclose(
            total_energy_traditional,
            total_energy_atomic_density,
            atol=accuracy_coarse,
        )
        assert np.isclose(
            total_energy_traditional,
            ldos_calculator.total_energy_dft_calculation,
            atol=accuracy_very_coarse,
        )

    @staticmethod
    def __simple_training(
        use_fast_tensor_dataset=False, use_openpmd_data=False
    ):
        """Perform a simple training and save it, if necessary."""
        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 400
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.data.use_fast_tensor_data_set = use_fast_tensor_dataset

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
        if use_openpmd_data:
            data_handler.add_snapshot(
                "Be_snapshot0.in.h5",
                data_path,
                "Be_snapshot0.out.h5",
                data_path,
                "tr",
                snapshot_type="openpmd",
            )
            data_handler.add_snapshot(
                "Be_snapshot1.in.h5",
                data_path,
                "Be_snapshot1.out.h5",
                data_path,
                "va",
                snapshot_type="openpmd",
            )
            data_handler.add_snapshot(
                "Be_snapshot2.in.h5",
                data_path,
                "Be_snapshot2.out.h5",
                data_path,
                "te",
                snapshot_type="openpmd",
            )
        else:
            data_handler.add_snapshot(
                "Be_snapshot0.in.npy",
                data_path,
                "Be_snapshot0.out.npy",
                data_path,
                "tr",
            )
            data_handler.add_snapshot(
                "Be_snapshot1.in.npy",
                data_path,
                "Be_snapshot1.out.npy",
                data_path,
                "va",
            )
            data_handler.add_snapshot(
                "Be_snapshot2.in.npy",
                data_path,
                "Be_snapshot2.out.npy",
                data_path,
                "te",
            )
        data_handler.prepare_data()

        # Train a network.
        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )
        test_trainer.train_network()

        return test_trainer
