import os

import mala
from mala import printout
from ase.io import read
from ase.md.nptberendsen import NVTBerendsen
from ase.md import MDLogger
from ase.calculators.espresso import Espresso
from ase.units import fs

from data_repo_path import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

"""
ex13_mala_md.py: Shows how MALA can be used to run a surrogate model based
MD simulation, and comares that to a QE calculation.
Please not that the resulting trajectories will NOT be correct. For one,
the QE MD will have far too few k-points to be meaningful. Secondly, 
for the MALA run, a non-local PP is being used, while the implementation
of HF forces in MALA relies on local PPs. Lastly, the network training is far
from rigorous here. 
For production runs, make sure to alleviate these issues by using an approriate 
amount of k-points, a local PP and a properly trained and verified network.  
"""


# Run a regular QE MD.
def md_qe():
    # Prepare the QE inputs for the run. This does not necessarily has
    # to be done with MALA.
    params = mala.Parameters()
    params.targets.pseudopotential_path = os.path.join(data_repo_path, "Be2")
    ldos = mala.LDOS(params)
    ldos.read_additional_calculation_data("qe.out", os.path.join(data_path, "Be_snapshot1.out"))
    ldos.qe_input_data["tprnfor"] = True

    # Set up the ASE objects.
    atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
    calculator = Espresso(input_data=ldos.qe_input_data,
                          pseudopotentials=ldos.qe_pseudopotentials)
    atoms.set_calculator(calculator)

    # Define and run the MD.
    dyn = NVTBerendsen(atoms, 1*fs, 298, 10)
    dyn.attach(MDLogger(dyn, atoms, 'qe_md.log', mode="w"), interval=1)
    dyn.run(10)


# Run a MALA MD.
def md_mala(network, new_parameters, iscaler, oscaler):
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
    new_parameters.targets.pseudopotential_path = os.path.join(data_repo_path, "Be2")

    # Construct a DataHandler object.
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Set up the ASE objects.
    atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
    calculator = mala.ASECalculator(new_parameters, network,
                                    inference_data_handler,
                                    reference_data=["qe.out", os.path.join(data_path, "Be_snapshot1.out")])
    atoms.set_calculator(calculator)

    # Define and run the MD.
    dyn = NVTBerendsen(atoms, 1*fs, 298, 10)
    dyn.attach(MDLogger(dyn, atoms, 'mala_md.log', mode="w"), interval=1)
    dyn.run(10)


# Trains a network.
def initial_training():
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
    test_parameters.data.data_splitting_snapshots = ["tr", "va"]

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
                              "Be_snapshot1.out.npy", data_path)
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path)
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


if __name__=="__main__":
    # First, train a MALA network.
    network, parameters, input_scaler, output_scaler = \
        initial_training()

    # Next, use these values to run an MD simulation.
    md_mala(network, parameters, input_scaler, output_scaler)

    # Finally, as an comparison, run a QE MD.
    md_qe()
