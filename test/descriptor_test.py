import importlib
import os

from ase.io import read
import mala
import numpy as np
import pytest

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

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
