import os

import mala
from mala import printout

from data_repo_path import data_repo_path
data_path = "/home/fiedlerl/jobs/qe/Be2/"

params_path = "./ex12_params.pkl"
network_path = "./ex12_network.pth"
input_scaler_path = "./ex12_iscaler.pkl"
output_scaler_path = "./ex12_oscaler.pkl"

# Uses a trained network to make a prediction.
def use_tester_class(network_path, params_path, input_scaler_path,
                        output_scaler_path):

    # First load Parameters and network.
    # Parameters may have to
    new_parameters = mala.Parameters.load_from_file(params_path,
                                                    no_snapshots=True)

    # Specify the correct LDOS parameters.
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5

    # Inference should ALWAYS be done with lazy loading activated, even if
    # training was not.
    new_parameters.data.use_lazy_loading = True

    # Load a network from a file.
    new_network = mala.Network.load_from_file(new_parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Add snapshots that are to be tested and make sure that the
    # data_splitting_snapshots list is correct.
    new_parameters.data.data_splitting_snapshots = ["te"]
    inference_data_handler.add_snapshot("Be_snapshot0.in.npy", data_path+"snap/",
                              "Be_snapshot2.out.npy", data_path+"ldos/",)
    inference_data_handler.prepare_data(reparametrize_scaler=False)

    # The Tester class is the testing analogon to the training class.
    tester = mala.Tester(new_parameters, new_network, inference_data_handler)

    # Get the results for the first (and only= snapshot.
    actual_ldos, predicted_ldos = tester.test_snapshot(0)

    # We will use the LDOS calculator to do some preprocessing.
    ldos_calculator = inference_data_handler.target_calculator
    ldos_calculator.read_additional_calculation_data("qe.out", os.path.join(
                                                     data_path,
                                                     "outputs/Be_snapshot2.out"))

    # Calculate the Band energy.
    band_energy_predicted = ldos_calculator.get_band_energy(predicted_ldos)
    band_energy_actual = ldos_calculator.get_band_energy(actual_ldos)
    printout("Band energy (actual, predicted, error)[eV]", band_energy_actual,
             band_energy_predicted, band_energy_predicted-band_energy_actual)

    # Calculate the number of electrons.
    nr_electrons_predicted = ldos_calculator.\
        get_number_of_electrons(predicted_ldos)
    nr_electrons_actual = ldos_calculator.get_number_of_electrons(actual_ldos)
    printout("Number of electrons (actual, predicted, error)[eV]",
             nr_electrons_actual, nr_electrons_predicted,
             nr_electrons_predicted-nr_electrons_actual)

def use_predictor(network_path, params_path, input_scaler_path,
                     output_scaler_path):
    # First load Parameters and network.
    # Parameters may have to
    new_parameters = mala.Parameters.load_from_file(params_path,
                                                    no_snapshots=True)

    # Specify the correct LDOS parameters.
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5
    new_parameters.running.inference_data_grid = [18, 18, 27]

    # Load a network from a file.
    new_network = mala.Network.load_from_file(new_parameters, network_path)

    # Make sure the same scaling is used for data handling.
    iscaler = mala.DataScaler.load_from_file(input_scaler_path)
    oscaler = mala.DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    predictor = mala.Predictor(new_parameters, new_network, inference_data_handler)
    ldos = predictor.predict_from_qeout(os.path.join(
                                                     data_path,
                                                     "outputs/Be_snapshot2.out"))
    inference_data_handler.target_calculator.read_additional_calculation_data("qe.out", os.path.join(
                                                     data_path,
                                                     "outputs/Be_snapshot2.out"))
    number_of_electrons = inference_data_handler.target_calculator.get_number_of_electrons(ldos)
    print(number_of_electrons)
    printout("Sucess!")


# Trains a network.
def initial_training(network_path, params_path, input_scaler_path,
                     output_scaler_path, desired_loss_improvement_factor=1):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.network.layer_activations = ["ReLU"]

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 400
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Be_snapshot0.in.npy", data_path+"snap/",
                              "Be_snapshot0.out.npy", data_path+"ldos/")
    data_handler.add_snapshot("Be_snapshot1.in.npy", data_path+"snap/",
                              "Be_snapshot1.out.npy", data_path+"ldos/")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path+"snap/",
                              "Be_snapshot2.out.npy", data_path+"ldos/",)
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                           100,
                                           data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    test_trainer.train_network()
    printout("Training: DONE.")

    ####################
    # SAVING
    # In order to be operational at a later point we need to save 4 objects:
    # Parameters, input/output scaler, network.
    ####################

    test_parameters.save(params_path)
    test_network.save_network(network_path)
    data_handler.input_data_scaler.save(input_scaler_path)
    data_handler.output_data_scaler.save(output_scaler_path)


# initial_training(network_path, params_path, input_scaler_path,
#                  output_scaler_path)
use_tester_class(network_path, params_path, input_scaler_path,
                    output_scaler_path)
use_predictor(network_path, params_path, input_scaler_path,
                    output_scaler_path)
