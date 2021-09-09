import os

import mala
import numpy as np

from data_repo_path import get_data_repo_path

data_path = os.path.join(get_data_repo_path(), "Al36/")

# Control how much the loss should be better after hyperopt compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 2

# Different HO methods will lead to different results, but they should be
# approximately the same.
desired_std_ho = 0.1


class TestHyperparameterOptimization:
    """Tests the basic hyperparameter optimization routines."""

    def test_hyperopt(self):
        """Test a regular, optuna based hyperparameter optimization."""

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
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
                                  "Al_debug_2k_nr0.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path,
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
        results = []
        result, last_study = self.__optimize_hyperparameters("optuna")
        results.append(result)
        results.append(self.__optimize_hyperparameters("oat"))
        results.append(self.
                       __optimize_hyperparameters("notraining",
                                                  input_creator_notraining=
                                                  "oat"))
        results.append(self.
                       __optimize_hyperparameters("notraining",
                                                  input_creator_notraining=
                                                  "optuna",
                                                  last_optuna_study=
                                                  last_study))
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
        test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
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
                                  "Al_debug_2k_nr0.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path,
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

    @staticmethod
    def __optimize_hyperparameters(hyper_optimizer,
                                   input_creator_notraining="oat",
                                   last_optuna_study=None):
        """Perform a hyperparameter optimization with the specified method."""

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
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
                                  "Al_debug_2k_nr0.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", data_path,
                                  "Al_debug_2k_nr1.out.npy", data_path,
                                  output_units="1/Ry")
        data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", data_path,
                                  "Al_debug_2k_nr2.out.npy", data_path,
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
        # If we do a notraining run currently we need to create an input array
        # using one of the other two possible hyperparameter optimizers.
        tmp_hp_optimizer = None
        if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
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
        elif hyper_optimizer == "notraining":
            tmp_parameters = test_parameters
            if input_creator_notraining == "optuna" and last_optuna_study is None:
                input_creator_notraining = "oat"
            tmp_parameters.hyperparameters.hyper_opt_method = \
                input_creator_notraining
            tmp_hp_optimizer = mala.HyperOptInterface(tmp_parameters,
                                                      data_handler)
            tmp_hp_optimizer.add_hyperparameter("categorical", "trainingtype",
                                                choices=["Adam", "SGD"])
            tmp_hp_optimizer.add_hyperparameter("categorical",
                                                "layer_activation_00",
                                                choices=["ReLU", "Sigmoid"])
            tmp_hp_optimizer.add_hyperparameter("categorical",
                                                "layer_activation_01",
                                                choices=["ReLU", "Sigmoid"])
            tmp_hp_optimizer.add_hyperparameter("categorical",
                                                "layer_activation_02",
                                                choices=["ReLU", "Sigmoid"])

        # Perform hyperparameter optimization.
        if hyper_optimizer == "oat" or hyper_optimizer == "optuna":
            test_hp_optimizer.perform_study()
            if hyper_optimizer == "optuna":
                last_optuna_study = test_hp_optimizer.get_trials_from_study()
        elif hyper_optimizer == "notraining":
            if input_creator_notraining == "optuna":
                test_hp_optimizer.perform_study(trial_list=last_optuna_study)
            else:
                test_hp_optimizer.perform_study(trial_list=
                                                tmp_hp_optimizer.get_orthogonal_array())
        test_hp_optimizer.set_optimal_parameters()

        # Train the final network.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()
        test_parameters.show()
        if hyper_optimizer == "optuna":
            return test_trainer.final_test_loss, last_optuna_study
        else:
            return test_trainer.final_test_loss
