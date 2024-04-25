import importlib
import os

import mala
import numpy as np
from ase.io import read
import pytest

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

# Control the various accuracies..
accuracy_snaps = 1e-4


class TestParallel:
    """Tests certain aspects of MALA's parallelization capabilities."""

    @pytest.mark.skipif(
        importlib.util.find_spec("lammps") is None,
        reason="LAMMPS is currently not part of the pipeline.",
    )
    def test_parallel_descriptors(self):
        """
        Test whether MALA can preprocess data.

        This means reading the LDOS from cube files and calculating
        bispectrum descriptors.
        The data necessary for this is currently not in the data repo!
        """

        # Set up parameters.
        test_parameters = mala.Parameters()
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 6
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.descriptors_contain_xyz = True

        atoms = read(os.path.join(data_path, "Be_snapshot1.out"))

        snap_calculator = mala.Bispectrum(test_parameters)
        snaps_serial, snapsize = snap_calculator.calculate_from_atoms(
            atoms, [18, 18, 27]
        )

        test_parameters = mala.Parameters()
        test_parameters.descriptors.descriptor_type = "Bispectrum"
        test_parameters.descriptors.bispectrum_twojmax = 6
        test_parameters.descriptors.bispectrum_cutoff = 4.67637
        test_parameters.descriptors.descriptors_contain_xyz = True
        test_parameters.descriptors.use_z_splitting = False
        test_parameters.use_mpi = True
        snap_calculator = mala.Bispectrum(test_parameters)
        snaps_parallel, snapsize = snap_calculator.calculate_from_atoms(
            atoms, [18, 18, 27]
        )
        snaps_parallel = snap_calculator.gather_descriptors(snaps_parallel)
        serial_shape = np.shape(snaps_serial)
        parallel_shape = np.shape(snaps_parallel)
        assert (
            serial_shape[0] == parallel_shape[0]
            and serial_shape[1] == parallel_shape[1]
            and serial_shape[2] == parallel_shape[2]
            and serial_shape[3] == parallel_shape[3]
        )
        assert np.isclose(
            np.sum(snaps_serial), np.sum(snaps_parallel), atol=accuracy_snaps
        )
