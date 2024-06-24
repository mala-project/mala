import os
import importlib
import sqlite3

import optuna

import mala
import numpy as np

from mala.datahandling.data_repo import data_path

# Control how much the loss should be better after hyperopt compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 2

# Different HO methods will lead to different results, but they should be
# approximately the same.
desired_std_ho = 0.1

# Values for the ACSD.
targeted_acsd_value = 0.04

# For the NASWOT comparison
# Since the values we are comparing here are dependent on the eigensolver
# implementation and numerical noise effects, I think it is reasonable
# to not require strict accuracy.
naswot_accuracy = 0.01


class TestHyperparameterOptimization:
    """Tests the basic hyperparameter optimization routines."""

    def test_hyperopt(self):
        """Test a regular, optuna based hyperparameter optimization."""

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.running.max_number_epochs = 20
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.hyperparameters.n_trials = 20
        test_parameters.hyperparameters.hyper_opt_method = "optuna"

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
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

        # Perform the hyperparameter optimization.
        test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)
        test_hp_optimizer.add_hyperparameter(
            "float", "learning_rate", 0.0000001, 0.01
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_00", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_01", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()

        #
        # To see if the hyperparameter optimization actually worked,
        # check if the best trial is better then the worst trial
        # by a certain factor.
        performed_trials_values = test_hp_optimizer.study.trials_dataframe()[
            "value"
        ]
        assert desired_loss_improvement_factor * min(
            performed_trials_values
        ) < max(performed_trials_values)

    def test_different_ho_methods(self):
        results = [
            self.__optimize_hyperparameters("optuna"),
            self.__optimize_hyperparameters("naswot"),
        ]

        # Since the OApackage is optional, we should only run
        # it if it is actually there.
        if importlib.util.find_spec("oapackage") is not None:
            results.append(self.__optimize_hyperparameters("oat"))

        assert np.std(results) < desired_std_ho

    def test_distributed_hyperopt(self):
        # Remove the previous data base.
        try:
            os.remove("test_ho.db")
        except FileNotFoundError:
            pass

        # Set up parameters
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.running.max_number_epochs = 5
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.hyperparameters.n_trials = 20
        test_parameters.hyperparameters.hyper_opt_method = "optuna"
        test_parameters.hyperparameters.study_name = "test_ho"
        test_parameters.hyperparameters.rdb_storage = "sqlite:///test_ho.db"

        # Load data
        data_handler = mala.DataHandler(test_parameters)

        # Add all the snapshots we want to use in to the list.
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

        # Create and perform hyperparameter optimization.
        test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)
        test_hp_optimizer.add_hyperparameter(
            "float", "learning_rate", 0.0000001, 0.01
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_00", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_01", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()
        performed_trials_values = test_hp_optimizer.study.trials_dataframe()[
            "value"
        ]
        assert desired_loss_improvement_factor * min(
            performed_trials_values
        ) < max(performed_trials_values)

    def test_acsd(self):
        """Test that the ACSD routine is still working."""
        test_parameters = mala.Parameters()

        test_parameters.descriptors.descriptors_contain_xyz = True
        test_parameters.descriptors.acsd_points = 100
        test_parameters.descriptors.descriptor_type = "Bispectrum"

        hyperoptimizer = mala.ACSDAnalyzer(test_parameters)
        hyperoptimizer.add_hyperparameter("bispectrum_twojmax", [2, 6])
        hyperoptimizer.add_hyperparameter("bispectrum_cutoff", [1.0])
        # hyperoptimizer.add_hyperparameter("bispectrum_twojmax", [6, 8])
        # hyperoptimizer.add_hyperparameter("bispectrum_cutoff", [1.0, 3.0])

        hyperoptimizer.add_snapshot(
            "espresso-out",
            os.path.join(data_path, "Be_snapshot1.out"),
            "numpy",
            os.path.join(data_path, "Be_snapshot1.in.npy"),
            target_units="1/(Ry*Bohr^3)",
        )
        hyperoptimizer.add_snapshot(
            "espresso-out",
            os.path.join(data_path, "Be_snapshot2.out"),
            "numpy",
            os.path.join(data_path, "Be_snapshot2.in.npy"),
            target_units="1/(Ry*Bohr^3)",
        )
        hyperoptimizer.perform_study()
        hyperoptimizer.set_optimal_parameters()

        # With these parameters, twojmax should always come out as 6.
        # Disabling for now, the small twojmax sometimesm lead to numerical
        # inconsistencies and since this is a part of the pipeline now
        # due to the python descriptors, this is more noticeable.
        # Will re-enable later, after Bartek and me (hot-)fix the ACSD.
        # assert hyperoptimizer.params.descriptors.bispectrum_twojmax == 6

    def test_naswot_eigenvalues(self):
        test_parameters = mala.Parameters()
        test_parameters.manual_seed = 1234
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.running.max_number_epochs = 10
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.hyperparameters.n_trials = 8
        test_parameters.hyperparameters.hyper_opt_method = "naswot"

        data_handler = mala.DataHandler(test_parameters)

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
        data_handler.prepare_data()

        test_hp_optimizer = mala.HyperOptNASWOT(test_parameters, data_handler)
        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            100,
            data_handler.output_dimension,
        ]
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.perform_study()
        correct_trial_list = [
            10569.71875,
            10649.0361328125,
            12081.2958984375,
            12360.3701171875,
            33523.9375,
            47565.8203125,
            149152.921875,
            150312.671875,
        ]
        for idx, trial in enumerate(correct_trial_list):
            assert np.isclose(
                trial,
                test_hp_optimizer.trial_losses[idx],
                rtol=naswot_accuracy,
            )

    @staticmethod
    def __optimize_hyperparameters(hyper_optimizer):
        """Perform a hyperparameter optimization with the specified method."""

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.running.max_number_epochs = 20
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.hyperparameters.n_trials = 8
        test_parameters.hyperparameters.hyper_opt_method = hyper_optimizer

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
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

        # Perform the actual hyperparameter optimization.
        test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)
        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            100,
            data_handler.output_dimension,
        ]

        # Add hyperparameters we want to have optimized to the list.
        # If we do a NASWOT run currently we can provide an input
        # array of trials.
        test_hp_optimizer.add_hyperparameter(
            "categorical", "optimizer", choices=["Adam", "SGD"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )

        # Perform hyperparameter optimization.
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()

        # Train the final network.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )
        test_trainer.train_network()
        test_parameters.show()
        return test_trainer.final_validation_loss

    def test_hyperopt_optuna_requeue_zombie_trials(self, tmp_path):

        ##tmp_path = os.environ["HOME"]

        db_filename = f"{tmp_path}/test_ho.db"

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.running.max_number_epochs = 2
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.hyperparameters.n_trials = 2
        test_parameters.hyperparameters.hyper_opt_method = "optuna"
        test_parameters.hyperparameters.study_name = "test_ho"
        test_parameters.hyperparameters.rdb_storage = (
            f"sqlite:///{db_filename}"
        )

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
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

        # Perform the hyperparameter optimization.
        test_hp_optimizer = mala.HyperOpt(test_parameters, data_handler)
        test_hp_optimizer.add_hyperparameter(
            "float", "learning_rate", 0.0000001, 0.01
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_00", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "int", "ff_neurons_layer_01", 10, 100
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_00", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_01", choices=["ReLU", "Sigmoid"]
        )
        test_hp_optimizer.add_hyperparameter(
            "categorical", "layer_activation_02", choices=["ReLU", "Sigmoid"]
        )

        def load_study():
            return optuna.load_study(
                study_name=test_parameters.hyperparameters.study_name,
                storage=test_parameters.hyperparameters.rdb_storage,
            )

        # First run, create database.
        test_hp_optimizer.perform_study()

        assert (
            test_hp_optimizer.study.trials_dataframe()["state"].to_list()
            == ["COMPLETE"] * 2
        )

        # This is basically the same code as in requeue_zombie_trials() but it
        # doesn't work. We get
        #   RuntimeError: Trial#0 has already finished and can not be updated.
        # This only works if state != COMPLETE, but this is what we have here.
        # So we need to hack the db directly.
        #
        ##study = load_study()
        ####study = test_hp_optimizer.study
        ##for trial in study.get_trials():
        ##    study._storage.set_trial_state_values(
        ##        trial_id=trial._trial_id, state=optuna.trial.TrialState.RUNNING
        ##    )

        con = sqlite3.connect(db_filename)
        cur = con.cursor()
        cur.execute("update trials set state='RUNNING'")
        con.commit()
        con.close()

        assert (
            load_study().trials_dataframe()["state"].to_list()
            == ["RUNNING"] * 2
        )

        test_hp_optimizer.requeue_zombie_trials(
            study_name=test_parameters.hyperparameters.study_name,
            rdb_storage=test_parameters.hyperparameters.rdb_storage,
        )
        assert (
            load_study().trials_dataframe()["state"].to_list()
            == ["WAITING"] * 2
        )

        # Second run adds one more trial.
        test_hp_optimizer.perform_study()
        assert (
            test_hp_optimizer.study.trials_dataframe()["state"].to_list()
            == ["COMPLETE"] * 3
        )
