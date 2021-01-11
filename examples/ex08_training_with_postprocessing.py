from fesl.common.parameters import Parameters
from fesl.datahandling.handler_interface import HandlerInterface
from fesl.datahandling.data_scaler import DataScaler
from fesl.network.network import Network
from fesl.network.trainer import Trainer
from fesl.targets.dos import DOS
import matplotlib.pyplot as plt
"""
ex08_training_with_postprocessing.py: Uses FESL to first train a network, use this network to predict the LDOS and then
analyze the results of this prediction. This example is structured a little bit different than other examples. 
It consists of two functions, to show that networks can be saved and reaccessed after being trained once.
By default, the training function is commented out. A saved network architecture and parameters is provided in
this repository, so this example focusses more on the analysis part. Nonetheless, the training can also be done 
by simply uncommenting the function call.
Please not that the values calculated at the end will be wrong, because we operate on 0.0025 % of a full DFT simulation
cell for this example. 
"""

# Uses a trained network to make a prediction.
def use_trained_network(network_path, params_path, input_scaler_path, output_scaler_path):

    # First we load Parameters and network.
    new_parameters = Parameters.load_from_file(params_path, no_snapshots=True)
    new_network = Network.load_from_file(new_parameters, network_path)

    # We use a data handler object to read the data we want to investigate.
    # We need to make sure that the same scaling is used.
    iscaler = DataScaler.load_from_file(input_scaler_path)
    oscaler = DataScaler.load_from_file(output_scaler_path)
    inference_data_handler = HandlerInterface(new_parameters, input_data_scaler=iscaler, output_data_scaler=oscaler)

    # Now we can add and load a snapshot to test our new data.
    # Note that we use prepare_data_for_inference instead of the regular prepare_data function.
    inference_data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")
    inference_data_handler.load_data()
    inference_data_handler.prepare_data_for_inference()

    # Now we can make a prediction.
    predicted_ldos = new_network.do_prediction(inference_data_handler.inference_data_inputs[0])
    actual_ldos = inference_data_handler.inference_data_outputs[0]

    # Now we use the prediction to calculate the band energy and compare it to the one we would get from the outputs themselves.
    predicted_ldos = oscaler.inverse_transform(predicted_ldos)
    actual_ldos = oscaler.inverse_transform(actual_ldos)
    predicted_ldos = predicted_ldos.detach().numpy()
    actual_ldos = actual_ldos.detach().numpy()

    # Use the LDOS object to do postprocessing.
    ldos_calculator = inference_data_handler.target_parser
    ldos_calculator.read_additional_calculation_data("qe.out", "./data/QE_Al.scf.pw.out")

    # Plot the DOS.
    actual_dos = ldos_calculator.get_density_of_states(actual_ldos)
    predicted_dos = ldos_calculator.get_density_of_states(predicted_ldos)
    plt.plot(actual_dos, label="actual")
    plt.plot(predicted_dos, label="predicted")
    plt.legend()
    plt.show()

    # Calculate the Band energy.
    # Use a DOS calculator to speed up processing.
    # This is important for bigger (actual) DOS arrays.
    dos_calculator = DOS.from_ldos(ldos_calculator)
    band_energy_predicted = dos_calculator.get_band_energy(predicted_dos)
    band_energy_actual = dos_calculator.get_band_energy(actual_dos)
    print("Band energy (actual, predicted, error)[eV]", band_energy_actual, band_energy_predicted, band_energy_predicted-band_energy_actual)


    nr_electrons_predicted = dos_calculator.get_number_of_electrons(predicted_dos)
    nr_electrons_actual = dos_calculator.get_number_of_electrons(actual_dos)
    print("Number of electrons (actual, predicted, error)[eV]", nr_electrons_actual, nr_electrons_predicted, nr_electrons_predicted-nr_electrons_actual)


# Trains a network.
def initial_training(network_path, params_path, input_scaler_path, output_scaler_path):
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that contains subclasses.
    ####################
    test_parameters = Parameters()
    test_parameters.data.datatype_in = "*.npy"
    test_parameters.data.datatype_out = "*.npy"
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va", "te"]
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.descriptors.twojmax = 11
    test_parameters.targets.ldos_gridsize = 10
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.training.max_number_epochs = 400
    test_parameters.training.mini_batch_size = 40
    test_parameters.training.learning_rate = 0.00001
    test_parameters.training.trainingtype = "Adam"
    test_parameters.targets.ldos_gridsize = 250
    test_parameters.targets.ldos_gridspacing_ev = 0.1
    test_parameters.targets.ldos_gridoffset_ev = -10

    ####################
    # DATA
    # Read data into RAM.
    # We have to specify the directories we want to read the snapshots from.
    # The Handlerinterface will also return input and output scaler objects. These are used internally to scale
    # the data. The objects can be used after successful training for inference or plotting.
    ####################

    data_handler = HandlerInterface(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Al_debug_2k_nr0.in.npy", "./data/", "Al_debug_2k_nr0.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr1.in.npy", "./data/", "Al_debug_2k_nr1.out.npy", "./data/", output_units="1/Ry")
    data_handler.add_snapshot("Al_debug_2k_nr2.in.npy", "./data/", "Al_debug_2k_nr2.out.npy", "./data/", output_units="1/Ry")

    data_handler.load_data()
    data_handler.prepare_data()
    print("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(), 100, data_handler.get_output_dimension()]

    # Setup network and trainer.
    test_network = Network(test_parameters)
    test_trainer = Trainer(test_parameters)
    print("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    print("Starting training.")
    test_trainer.train_network(test_network, data_handler)
    print("Training: DONE.")

    ####################
    # SAVING
    # In order to be operational at a later point we need to save 4 objects: Parameters, input/output scaler, network.
    ####################

    test_parameters.save(params_path)
    test_network.save_network(network_path)
    data_handler.input_data_scaler.save(input_scaler_path)
    data_handler.output_data_scaler.save(output_scaler_path)

print("Welcome to FESL.")
print("Running ex08_training_with_postprocessing.py")

# Choose the paths where the network and the parameters for it should be saved.
params_path = "./data/ex08_params.pkl"
network_path = "./data/ex08_network.pth"
input_scaler_path = "./data/ex08_iscaler.pkl"
output_scaler_path = "./data/ex08_oscaler.pkl"

# Training was already done, but can be run again by uncommenting the next line.
# initial_training(network_path, params_path, input_scaler_path, output_scaler_path)
# Prediction and analysis using an already trained network.
use_trained_network(network_path, params_path, input_scaler_path, output_scaler_path)

print("Successfully ran ex08_training_with_postprocessing.py.")
