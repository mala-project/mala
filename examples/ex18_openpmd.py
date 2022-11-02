import os

import mala

"""
Development example for OpenPMD.

Requires LAMMPS interface and some additional data.
"""

# This directory contains actual simulation data which we can use.
full_data_path = "/home/fiedlerl/jobs/qe/Be2/07_dft"

# This directory will temporarily hold the training data we create
# for development.
data_path = "./training_data_temp"


# Define the MALA parameters for this run.
parameters = mala.Parameters()
# Debug output.
parameters.verbosity = 2

# SNAP calculation.
parameters.descriptors.descriptor_type = "SNAP"
parameters.descriptors.twojmax = 6
parameters.descriptors.rcutfac = 4.67637
parameters.descriptors.descriptors_contain_xyz = True

# LDOS parameters.
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

# Useful if we want to compare numpy as openpmd.
parameters.manual_seed = 1234

# Network (training) parameters.
parameters.data.data_splitting_type = "by_snapshot"
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
parameters.network.layer_activations = ["ReLU"]
parameters.running.max_number_epochs = 400
parameters.running.mini_batch_size = 40
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"


def convert_data():
    """Convert/calculate volumetric data to numpy/hdf5 for later use."""
    # Prepare 4 snapshots for conversion.
    data_converter = mala.DataConverter(parameters)
    for i in range(0, 4):
        data_converter.\
            add_snapshot_qeout_cube(os.path.join(full_data_path,
                                                 "snapshot"+str(i)
                                                 + "/snapshot"+str(i)+".out"),
                                    os.path.join(full_data_path,
                                                 "snapshot"+str(i)
                                                 + "/tmp.pp0*Be_snapshot"+str(i)
                                                 + "_ldos.cube"),
                                    output_units="1/(Ry*Bohr^3)")

    # Convert data both as OpenPMD HDF5 (default) and numpy.
    data_converter.convert_snapshots("./training_data_temp",
                                     naming_scheme="Be_snapshot*")
    data_converter.convert_snapshots("./training_data_temp",
                                     naming_scheme="Be_snapshot*",
                                     use_numpy=True)


def conversion_test():
    """Test whether numpy/hdf5 yield comparable LDOS."""
    ldos_calculator = mala.LDOS.\
        from_hdf5_file(parameters, "training_data_temp/Be_snapshot0.out.h5")
    ldos_calculator2 = mala.LDOS.\
        from_numpy_file(parameters, "training_data_temp/Be_snapshot0.out.npy")

    ldos_calculator.\
        read_additional_calculation_data("qe.out",
                                         data=os.path.join("training_data_temp",
                                                           "snapshot0.out"))
    ldos_calculator2.\
        read_additional_calculation_data("qe.out",
                                         data=os.path.join("training_data_temp",
                                                           "snapshot0.out"))

    # These two values should match.
    print(ldos_calculator.band_energy)
    print(ldos_calculator2.band_energy)


def network_training_test(use_numpy=False):
    """Test whether a network can be trained via OpenPMD/numpy data."""

    # Load data.
    data_handler = mala.DataHandler(parameters)
    data_handler.clear_data()
    if use_numpy:
        data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                  "Be_snapshot0.out.npy", data_path, "tr",
                                  snapshot_type="numpy")
        data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                  "Be_snapshot1.out.npy", data_path, "va",
                                  snapshot_type="numpy")
    else:
        data_handler.add_snapshot("Be_snapshot0.in.h5", data_path,
                                  "Be_snapshot0.out.h5", data_path, "tr",
                                  snapshot_type="hdf5")
        data_handler.add_snapshot("Be_snapshot1.in.h5", data_path,
                                  "Be_snapshot1.out.h5", data_path, "va",
                                  snapshot_type="hdf5")
    data_handler.prepare_data()

    # Prepare and train network.
    parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                      100,
                                      data_handler.get_output_dimension()]

    # Reducing output for training.
    parameters.verbosity = 0
    test_network = mala.Network(parameters)
    test_trainer = mala.Trainer(parameters, test_network, data_handler)
    test_trainer.train_network()
    parameters.verbosity = 2

    # Test the model.

    # First, load data for testing.
    parameters.data.use_lazy_loading = True
    data_handler.clear_data()
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path,
                              "te", snapshot_type="numpy")
    data_handler.prepare_data(reparametrize_scaler=False)
    tester = mala.Tester(parameters, test_network, data_handler)

    # Get a model prediction.
    actual_ldos, predicted_ldos = tester.test_snapshot(0)

    # Use the prediction to calculate some numbers that can be easily
    # compared.
    ldos_calculator: mala.LDOS
    ldos_calculator = data_handler.target_calculator
    ldos_calculator.read_additional_calculation_data("qe.out", os.path.join(
        data_path,
        "snapshot2.out"))

    # Predicted and actual should be close to one another, and the same
    # whether numpy or hdf5 has been used.
    ldos_calculator.read_from_array(predicted_ldos)
    print("Predicted: ", ldos_calculator.band_energy,
          ldos_calculator.number_of_electrons)
    ldos_calculator.read_from_array(actual_ldos)
    print("Actual: ", ldos_calculator.band_energy,
          ldos_calculator.number_of_electrons)
    parameters.data.use_lazy_loading = False


convert_data()
conversion_test()

# Note: due to the way the random seed works, these functions will only
# generate the exact same result when called one at a time.
network_training_test(use_numpy=False)
network_training_test(use_numpy=True)
