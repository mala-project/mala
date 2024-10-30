import importlib
import os

from ase.io import read
import mala
from mala.common.parameters import ParametersBase
import numpy as np
import pytest


from mala.datahandling.data_repo import data_path


# This test checks whether MALA interfaces to other codes, mainly the ASE
# calculator, still work.

# For the ASE calculator test, it's enough when the energies are roughly the
# same.
accuracy_coarse = 10

# The JSON additional data test further down implies a high accuracy.
accuracy_fine = 1e-16


class TestInterfaces:
    """Tests MALA interfaces."""

    def test_json(self):
        """
        Test whether MALA JSON interface is still working.

        Please note that this does not test whether all parameters are
        correctly serializable, only the interface itself.
        """
        params = mala.Parameters()
        # Change a few parameter to see if anything is actually happening.
        params.manual_seed = 2022
        params.network.layer_sizes = [100, 100, 100]
        params.network.layer_activations = ["test", "test"]
        params.descriptors.bispectrum_cutoff = 4.67637

        # Save, load, compare.
        params.save("interface_test.json")
        new_params = params.load_from_file("interface_test.json")
        for v in vars(params):
            if isinstance(getattr(params, v), ParametersBase):
                v_old = getattr(params, v)
                v_new = getattr(new_params, v)
                for subv in vars(v_old):
                    assert getattr(v_new, subv) == getattr(v_old, subv)
            else:
                assert getattr(new_params, v) == getattr(params, v)

    @pytest.mark.skipif(
        importlib.util.find_spec("openpmd_api") is None,
        reason="No OpenPMD found on this machine, skipping " "test.",
    )
    def test_openpmd_io(self):
        params = mala.Parameters()

        # Read an LDOS and some additional data for it.
        ldos_calculator = mala.LDOS.from_numpy_file(
            params, os.path.join(data_path, "Be_snapshot1.out.npy")
        )
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot1.out"), "espresso-out"
        )

        # Write and then read in via OpenPMD and make sure all the info is
        # retained.
        ldos_calculator.write_to_openpmd_file(
            "test_openpmd.h5", ldos_calculator.local_density_of_states
        )
        ldos_calculator2 = mala.LDOS.from_openpmd_file(
            params, "test_openpmd.h5"
        )

        assert np.isclose(
            np.sum(
                ldos_calculator.local_density_of_states
                - ldos_calculator.local_density_of_states
            ),
            0.0,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.fermi_energy_dft,
            ldos_calculator2.fermi_energy_dft,
            rtol=accuracy_fine,
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("total_energy") is None
        or importlib.util.find_spec("lammps") is None,
        reason="QE and LAMMPS are currently not part of the " "pipeline.",
    )
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
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.optimizer = "Adam"
        test_parameters.targets.target_type = "LDOS"
        test_parameters.targets.ldos_gridsize = 11
        test_parameters.targets.ldos_gridspacing_ev = 2.5
        test_parameters.targets.ldos_gridoffset_ev = -5
        test_parameters.running.inference_data_grid = [18, 18, 27]
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 10
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.targets.pseudopotential_path = data_path

        ####################
        # DATA
        ####################

        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot(
            "Be_snapshot1.in.npy",
            data_path,
            "Be_snapshot1.out.npy",
            data_path,
            "tr",
        )
        data_handler.add_snapshot(
            "Be_snapshot2.in.npy",
            data_path,
            "Be_snapshot2.out.npy",
            data_path,
            "va",
        )
        data_handler.prepare_data()

        ####################
        # NETWORK SETUP AND TRAINING.
        ####################

        test_parameters.network.layer_sizes = [
            data_handler.input_dimension,
            100,
            data_handler.output_dimension,
        ]

        # Setup network and trainer.
        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(
            test_parameters, test_network, data_handler
        )
        test_trainer.train_network()

        ####################
        # INTERFACING.
        ####################

        # Set up the ASE objects.
        atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
        calculator = mala.MALA(
            test_parameters,
            test_network,
            data_handler,
            reference_data=os.path.join(data_path, "Be_snapshot1.out"),
        )
        total_energy_dft_calculation = (
            calculator.data_handler.target_calculator.total_energy_dft_calculation
        )
        calculator.calculate(atoms, properties=["energy"])
        assert np.isclose(
            total_energy_dft_calculation,
            calculator.results["energy"],
            atol=accuracy_coarse,
        )

    def test_additional_calculation_data_json(self):
        test_parameters = mala.Parameters()
        ldos_calculator = mala.LDOS(test_parameters)
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path, "Be_snapshot1.out"), "espresso-out"
        )
        ldos_calculator.write_additional_calculation_data(
            "additional_calculation_data.json"
        )
        new_ldos_calculator = mala.LDOS(test_parameters)
        new_ldos_calculator.read_additional_calculation_data(
            "additional_calculation_data.json", "json"
        )

        # Verify that essentially the same info has been loaded.
        assert np.isclose(
            ldos_calculator.fermi_energy_dft,
            new_ldos_calculator.fermi_energy_dft,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.temperature,
            new_ldos_calculator.temperature,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.number_of_electrons_exact,
            new_ldos_calculator.number_of_electrons_exact,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.band_energy_dft_calculation,
            new_ldos_calculator.band_energy_dft_calculation,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.total_energy_dft_calculation,
            new_ldos_calculator.total_energy_dft_calculation,
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.number_of_electrons_from_eigenvals,
            new_ldos_calculator.number_of_electrons_from_eigenvals,
            rtol=accuracy_fine,
        )
        assert (
            ldos_calculator.qe_input_data["ibrav"]
            == new_ldos_calculator.qe_input_data["ibrav"]
        )
        assert np.isclose(
            ldos_calculator.qe_input_data["ecutwfc"],
            new_ldos_calculator.qe_input_data["ecutwfc"],
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.qe_input_data["ecutrho"],
            new_ldos_calculator.qe_input_data["ecutrho"],
            rtol=accuracy_fine,
        )
        assert np.isclose(
            ldos_calculator.qe_input_data["degauss"],
            new_ldos_calculator.qe_input_data["degauss"],
            rtol=accuracy_fine,
        )
        for key in ldos_calculator.qe_pseudopotentials.keys():
            assert (
                new_ldos_calculator.qe_pseudopotentials[key]
                == ldos_calculator.qe_pseudopotentials[key]
            )
        for i in range(0, 3):
            assert (
                ldos_calculator.grid_dimensions[i]
                == new_ldos_calculator.grid_dimensions[i]
            )
            assert (
                ldos_calculator.atoms.pbc[i]
                == new_ldos_calculator.atoms.pbc[i]
            )

            for j in range(0, 3):
                assert np.isclose(
                    ldos_calculator.voxel[i, j],
                    new_ldos_calculator.voxel[i, j],
                )
                assert np.isclose(
                    ldos_calculator.atoms.get_cell()[i, j],
                    new_ldos_calculator.atoms.get_cell()[i, j],
                    rtol=accuracy_fine,
                )

        for i in range(0, len(ldos_calculator.atoms)):
            for j in range(0, 3):
                assert np.isclose(
                    ldos_calculator.atoms.get_positions()[i, j],
                    new_ldos_calculator.atoms.get_positions()[i, j],
                    rtol=accuracy_fine,
                )
