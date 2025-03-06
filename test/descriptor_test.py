import importlib
import os

from ase.io import read
import mala
import numpy as np
import pytest

from mala.datahandling.data_repo import data_path

# Accuracy of test.
accuracy_descriptors = 5e-8


class TestDescriptorImplementation:
    """Tests the MALA python based descriptor implementation against LAMMPS."""

    @pytest.mark.skipif(
        importlib.util.find_spec("lammps") is None,
        reason="LAMMPS is currently not part of the pipeline.",
    )
    def test_bispectrum(self):
        """Calculate bispectrum descriptors with LAMMPS / MALA and compare."""
        params = mala.Parameters()
        params.descriptors.bispectrum_cutoff = 4.67637
        params.descriptors.bispectrum_twojmax = 4

        bispectrum_calculator = mala.descriptors.Bispectrum(params)
        atoms = read(os.path.join(data_path, "Be_snapshot3.out"))

        descriptors, ngrid = bispectrum_calculator.calculate_from_atoms(
            atoms=atoms, grid_dimensions=[18, 18, 27]
        )
        params.use_lammps = False
        descriptors_py, ngrid = bispectrum_calculator.calculate_from_atoms(
            atoms=atoms, grid_dimensions=[18, 18, 27]
        )

        assert (
            np.abs(
                np.mean(
                    descriptors_py[:, :, :, 0:3] - descriptors[:, :, :, 0:3]
                )
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.mean(descriptors_py[:, :, :, 3] - descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.std(descriptors_py[:, :, :, 3] / descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("lammps") is None,
        reason="LAMMPS is currently not part of the pipeline.",
    )
    def test_gaussian(self):
        """Calculate bispectrum descriptors with LAMMPS / MALA and compare."""
        params = mala.Parameters()
        params.descriptors.atomic_density_cutoff = 4.67637

        bispectrum_calculator = mala.descriptors.AtomicDensity(params)
        atoms = read(os.path.join(data_path, "Be_snapshot3.out"))

        descriptors, ngrid = bispectrum_calculator.calculate_from_atoms(
            atoms=atoms, grid_dimensions=[18, 18, 27]
        )
        params.use_lammps = False
        descriptors_py, ngrid = bispectrum_calculator.calculate_from_atoms(
            atoms=atoms, grid_dimensions=[18, 18, 27]
        )

        assert (
            np.abs(
                np.mean(
                    descriptors_py[:, :, :, 0:3] - descriptors[:, :, :, 0:3]
                )
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.mean(descriptors_py[:, :, :, 3] - descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.std(descriptors_py[:, :, :, 3] / descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )

    @pytest.mark.skipif(
        importlib.util.find_spec("lammps") is None,
        reason="LAMMPS is currently not part of the pipeline.",
    )
    def test_ace(self):
        """Calculate ACE descriptors and check nothing has changed."""
        params = mala.Parameters()
        params.descriptors.descriptor_type = "ACE"
        params.descriptors.ace_cutoff_factor = 5.8
        params.descriptors.ace_included_expansion_ranks = [1, 2, 3]
        params.descriptors.ace_maximum_l_per_rank = [0, 1, 1]
        params.descriptors.ace_maximum_n_per_rank = [1, 1, 1]
        params.descriptors.ace_minimum_l_per_rank = [0, 0, 0]

        ace_calculator = mala.ACE(params)
        atoms = read(os.path.join(data_path, "Be_snapshot0.out"))

        descriptors, ngrid = ace_calculator.calculate_from_atoms(
            atoms=atoms, grid_dimensions=[18, 18, 27]
        )
        ace_calculator2 = mala.ACE(params)

        # Disabling coordinate cutting because we want to compare raw data.
        params.descriptors.descriptors_contain_xyz = False
        descriptors_old = ace_calculator2.read_from_numpy_file(
            os.path.join(data_path, "Be_snapshot0_ACE.in.npy")
        )
        assert (
            np.abs(
                np.mean(
                    descriptors_old[:, :, :, 0:3] - descriptors[:, :, :, 0:3]
                )
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.mean(descriptors_old[:, :, :, 3] - descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )
        assert (
            np.abs(
                np.std(descriptors_old[:, :, :, 3] / descriptors[:, :, :, 3])
            )
            < accuracy_descriptors
        )
