import importlib
import os

from ase.io import read
import mala
import numpy as np
import pytest


from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")


# This test checks whether MALA interfaces to other codes, mainly the ASE
# calculator, still work.

# For the ASE calculator test, it's enough when the energies are roughly the
# same.
accuracy_coarse = 10


class TestInterfaces:
    """Tests MALA interfaces."""

    @pytest.mark.skipif(importlib.util.find_spec("lammps") is None,
                        reason="LAMMPS is currently not part of the pipeline.")
    def test_ase_calculator(self):
        """
        Test whether the ASE calculator class can still be used.

        This test tests for serial and energy calculation only.
        Forces are still an experimental feature, so they are not included
        here.
        """
        ####################
        # PARAMETERS
        ####################

        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.data_splitting_snapshots = ["tr", "va"]
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.running.inference_data_grid = [18, 18, 27]
        test_parameters.descriptors.descriptor_type = "SNAP"
        test_parameters.descriptors.twojmax = 10
        test_parameters.descriptors.rcutfac = 4.67637
        test_parameters.targets.pseudopotential_path = os.path.join(
            data_repo_path,
            "Be2")

        ####################
        # DATA
        ####################

        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                  "Be_snapshot1.out.npy", data_path)
        data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                                  "Be_snapshot2.out.npy", data_path)
        data_handler.prepare_data()

        ####################
        # NETWORK SETUP AND TRAINING.
        ####################

        test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                               100,
                                               data_handler.get_output_dimension()]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
        test_trainer.train_network()

        ####################
        # INTERFACING.
        ####################

        # Set up the ASE objects.
        atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
        calculator = mala.ASECalculator(test_parameters, test_network,
                                        data_handler,
                                        reference_data=
                                        ["qe.out",
                                         os.path.join(data_path,
                                                      "Be_snapshot1.out")])
        total_energy_dft_calculation = calculator.data_handler.\
            target_calculator.total_energy_dft_calculation
        calculator.calculate(atoms, properties=["energy"])
        assert np.isclose(total_energy_dft_calculation,
                          calculator.results["energy"],
                          atol=accuracy_coarse)
