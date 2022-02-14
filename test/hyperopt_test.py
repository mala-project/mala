import os

import mala
import numpy as np

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Al36")
data_path_be = os.path.join(os.path.join(data_repo_path, "Be2"),
                            "training_data")

# Control how much the loss should be better after hyperopt compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 2

# Different HO methods will lead to different results, but they should be
# approximately the same.
desired_std_ho = 0.1

# Values for the ACSD.
targeted_acsd_value = 0.04


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
        test_parameters.running.trainingtype = "Adam"
        test_parameters.hyperparameters.n_trials = 20
        test_parameters.hyperparameters.hyper_opt_method = "optuna"

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                  "Al_debug_2k_nr0.out.npy", data_path, "tr",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path, "va",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path, "te",
                                  output_units="1/Ry")
        data_handler.prepare_data()

        # Perform the hyperparameter optimization.
        test_hp_optimizer = mala.HyperOptInterface(test_parameters,
                                                   data_handler)
        test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                             0.0000001, 0.01)
        test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10,
                                             100)
        test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10,
                                             100)
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_00",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_01",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_02",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()

        #
        # To see if the hyperparameter optimization actually worked,
        # check if the best trial is better then the worst trial
        # by a certain factor.
        performed_trials_values = test_hp_optimizer.study. \
                                  trials_dataframe()["value"]
        assert desired_loss_improvement_factor * \
               min(performed_trials_values) < \
               max(performed_trials_values)

    def test_different_ho_methods(self):
        results = [self.__optimize_hyperparameters("optuna"),
                   self.__optimize_hyperparameters("oat"),
                   self.__optimize_hyperparameters("naswot")]

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
        test_parameters.running.trainingtype = "Adam"
        test_parameters.hyperparameters.n_trials = 20
        test_parameters.hyperparameters.hyper_opt_method = "optuna"
        test_parameters.hyperparameters.study_name = "test_ho"
        test_parameters.hyperparameters.rdb_storage = 'sqlite:///test_ho.db'

        # Load data
        data_handler = mala.DataHandler(test_parameters)

        # Add all the snapshots we want to use in to the list.
        data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                  "Al_debug_2k_nr0.out.npy", data_path, "tr",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path, "va",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path, "te",
                                  output_units="1/Ry")
        data_handler.prepare_data()

        # Create and perform hyperparameter optimization.
        test_hp_optimizer = mala.HyperOptInterface(test_parameters,
                                                   data_handler)
        test_hp_optimizer.add_hyperparameter("float", "learning_rate",
                                             0.0000001, 0.01)
        test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_00", 10,
                                             100)
        test_hp_optimizer.add_hyperparameter("int", "ff_neurons_layer_01", 10,
                                             100)
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_00",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_01",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_02",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()
        performed_trials_values = test_hp_optimizer.study. \
                                  trials_dataframe()["value"]
        assert desired_loss_improvement_factor * \
               min(performed_trials_values) < \
               max(performed_trials_values)

    def test_acsd(self):
        """Test that the ACSD routine is still working."""
        test_parameters = mala.Parameters()
        test_parameters.descriptors.acsd_points = 100
        descriptors = mala.DescriptorInterface(test_parameters)
        snap_data = np.load(os.path.join(data_path_be, "Be_snapshot1.in.npy"))
        ldos_data = np.load(os.path.join(data_path_be, "Be_snapshot1.out.npy"))
        assert descriptors.get_acsd(snap_data, ldos_data) < targeted_acsd_value

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
        test_parameters.running.trainingtype = "Adam"
        test_parameters.hyperparameters.n_trials = 8
        test_parameters.hyperparameters.hyper_opt_method = hyper_optimizer

        # Load data.
        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", data_path,
                                  "Al_debug_2k_nr0.out.npy", data_path, "tr",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path, "va",
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path, "te",
                                  output_units="1/Ry")
        data_handler.prepare_data()

        # Perform the actual hyperparameter optimization.
        test_hp_optimizer = mala.HyperOptInterface(test_parameters,
                                                   data_handler)
        test_parameters.network.layer_sizes = [
            data_handler.get_input_dimension(),
            100, 100,
            data_handler.get_output_dimension()]

        # Add hyperparameters we want to have optimized to the list.
        # If we do a NASWOT run currently we can provide an input
        # array of trials.
        test_hp_optimizer.add_hyperparameter("categorical", "trainingtype",
                                             choices=["Adam", "SGD"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_00",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_01",
                                             choices=["ReLU", "Sigmoid"])
        test_hp_optimizer.add_hyperparameter("categorical",
                                             "layer_activation_02",
                                             choices=["ReLU", "Sigmoid"])

        # Perform hyperparameter optimization.
        test_hp_optimizer.perform_study()
        test_hp_optimizer.set_optimal_parameters()

        # Train the final network.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()
        test_parameters.show()
        return test_trainer.final_test_loss
