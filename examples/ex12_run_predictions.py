import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

"""
ex12_run_predictions.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
def use_predictor(new_network, new_parameters, iscaler, oscaler):
    ####################
    # PARAMETERS
    # Specify the correct LDOS parameters and load network.
    ####################
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5
    new_parameters.running.inference_data_grid = [18, 18, 27]

    new_parameters.descriptors.descriptor_type = "SNAP"
    new_parameters.descriptors.twojmax = 10
    new_parameters.descriptors.rcutfac = 4.67637

    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    ####################
    # PREDICTION
    # Create a Predictor object and make a prediction.
    ####################

    predictor = mala.Predictor(new_parameters, new_network,
                               inference_data_handler)

    # Using a QE out is simply for convenience; with the function
    # .predict_from_atoms, prediction can be made directly from atomic
    # configurations.
    ldos = predictor.predict_from_qeout(os.path.join(
                                                     data_path,
                                                     "Be_snapshot3.out"))
    ldos_calculator: mala.LDOS = inference_data_handler.target_calculator

    # Fermi energy has to be calculated self-consistently for predictions,
    # and for that we also have to tell the calculator how many electrons
    # the system has and at which temperature we perform calculations.
    ldos_calculator.number_of_electrons_exact = 4
    ldos_calculator.temperature_K = 298

    fermi_energy = ldos_calculator.get_self_consistent_fermi_energy_ev(ldos)

    number_of_electrons = ldos_calculator.\
        get_number_of_electrons(ldos, fermi_energy_eV=fermi_energy)
    band_energy = ldos_calculator. \
        get_band_energy(ldos, fermi_energy_eV=fermi_energy)
    printout("Predicted number of electrons: ", number_of_electrons)
    printout("Predicted band energy: ", band_energy)


# Trains a network.
def initial_training():
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis.
    test_parameters.data.data_splitting_type = "by_snapshot"

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.network.layer_activations = ["ReLU"]

    # Specify the training parameters.
    test_parameters.running.max_number_epochs = 100
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                              "Be_snapshot1.out.npy", data_path, "tr",
                              output_units="1/(eV*Bohr^3)")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path, "va",
                              output_units="1/(eV*Bohr^3)")
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

    return test_network, test_parameters, data_handler.input_data_scaler,\
           data_handler.output_data_scaler


network, params, input_scaler, output_scaler = initial_training()
use_predictor(network, params, input_scaler, output_scaler)
