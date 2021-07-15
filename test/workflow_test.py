import mala
from data_repo_path import get_data_repo_path
data_path = get_data_repo_path()+"Al36/"

# Control how much the loss should be better after training compared to
# before. This value is fairly high, but we're training on absolutely
# minimal amounts of data.
desired_loss_improvement_factor = 1


class TestFullWorkflow:
    """Tests an entire MALA workflow."""

    def test_network_training(self):
        """Test whether MALA can train a NN."""

        # Set up parameters.

        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 20
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"

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

        # Buld and train a networ.

        test_parameters.network.layer_sizes = [
            data_handler.get_input_dimension(),
            100,
            data_handler.get_output_dimension()]

        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()

        assert desired_loss_improvement_factor * \
               test_trainer.initial_test_loss > test_trainer.final_test_loss
