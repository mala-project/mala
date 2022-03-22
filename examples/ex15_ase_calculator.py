import os

import mala
from mala import printout
from ase.io import read

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

"""
ex15_ase_calculator.py: Shows how MALA can be used as an ASE calculator. 
Currently, calculation of forces is not supported. 

REQUIRES LAMMPS AND QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""


# Use an ASE calculator to make predictions.
def use_calculator(network, new_parameters, iscaler, oscaler):
    ####################
    # PARAMETERS
    # Make sure all parameters are prepared for the run.
    ####################
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5
    new_parameters.running.inference_data_grid = [18, 18, 27]

    new_parameters.descriptors.descriptor_type = "SNAP"
    new_parameters.descriptors.twojmax = 10
    new_parameters.descriptors.rcutfac = 4.67637
    new_parameters.targets.pseudopotential_path = os.path.join(data_repo_path,
                                                               "Be2")

    # Construct a DataHandler object.
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Set up the ASE objects.
    atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
    calculator = mala.MALA(new_parameters, network,
                           inference_data_handler,
                           reference_data=
                                    ["qe.out",
                                     os.path.join(data_path,
                                                  "Be_snapshot1.out")])
    atoms.set_calculator(calculator)
    print(atoms.get_potential_energy())


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
                              "Be_snapshot1.out.npy", data_path, "tr")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path, "va")
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
    # Save networks for debugging.
    test_network.save_network("ex15.network.pth")
    test_parameters.save("ex15.params.json")
    data_handler.input_data_scaler.save("ex15.iscaler.pkl")
    data_handler.output_data_scaler.save("ex15.oscaler.pkl")
    return test_network, test_parameters, data_handler.input_data_scaler,\
           data_handler.output_data_scaler


if __name__ == "__main__":
    # First, train a MALA network.
    network, parameters, input_scaler, output_scaler = \
        initial_training()

    # Instead of training, these can also be loaded.
    # parameters = mala.Parameters.load_from_file("ex15.params.json")
    # input_scaler = mala.DataScaler.load_from_file("ex15.iscaler.pkl")
    # output_scaler = mala.DataScaler.load_from_file("ex15.oscaler.pkl")
    # network = mala.Network.load_from_file(parameters, "ex15.network.pth")

    # Next, use these values to run an MD simulation.
    use_calculator(network, parameters, input_scaler, output_scaler)

