import os

import numpy as np

import mala
from mala.datahandling.data_repo import data_path_be, data_path_bao

accuracy_band_energy = 1
accuracy_strict = 1e-16


class TestSplitting:
    """
    Class that tests that splitted (L)DOS data can be processed

    Splitted means along the energy domain.
    """

    def test_ldos_splitting(self):
        parameters3 = mala.Parameters()
        parameters3.targets.ldos_gridsize = [
            3,
            3,
            7,
        ]
        parameters3.targets.ldos_gridoffset_ev = [-5, 0, 5]
        parameters3.targets.ldos_gridspacing_ev = [2.5, 2.5, 2.5]
        cubes_path = os.path.join(data_path_be, "cubes")
        ldos_calculator3 = mala.LDOS(parameters3)
        ldos_calculator3.read_from_cube(
            [
                os.path.join(cubes_path, "tmp.pp0*Be_part1_ldos.cube"),
                os.path.join(cubes_path, "tmp.pp0*Be_part2_ldos.cube"),
                os.path.join(cubes_path, "tmp.pp0*Be_part3_ldos.cube"),
            ]
        )

        parameters = mala.Parameters()
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5
        parameters.targets.ldos_gridoffset_ev = -5
        ldos_calculator = mala.LDOS(parameters)
        ldos_calculator.read_from_cube(
            os.path.join(cubes_path, "tmp.pp*Be_ldos.cube")
        )

        print(
            np.mean(
                np.abs(
                    ldos_calculator3.local_density_of_states
                    - ldos_calculator.local_density_of_states
                )
            )
        )

    def test_dos_splitting(self):
        parameters3 = mala.Parameters()
        parameters3.targets.ldos_gridsize = [
            3,
            3,
            7,
        ]
        parameters3.targets.ldos_gridoffset_ev = [-5, 0, 5]
        parameters3.targets.ldos_gridspacing_ev = [2.5, 2.5, 2.5]
        cubes_path = os.path.join(data_path_be, "cubes")
        dos_calculator3 = mala.DOS(parameters3)
        dos_calculator3.read_from_qe_out(
            os.path.join(data_path_be, "Be_snapshot1.out"),
            smearing_factor=[2, 2, 2],
        )

        parameters = mala.Parameters()
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5
        parameters.targets.ldos_gridoffset_ev = -5
        dos_calculator = mala.DOS(parameters)
        dos_calculator.read_from_qe_out(
            os.path.join(data_path_be, "Be_snapshot1.out")
        )

        print(
            np.mean(
                np.abs(
                    dos_calculator3.density_of_states
                    - dos_calculator.density_of_states
                )
            )
        )

    # This test is based on examples/advanced/ex09_align_ldos.py, but adds a
    # test to check that the aligned data is equivalent between the numpy-based
    # and the openPMD-based implementations.
    def test_ldos_alignment(self):
        parameters = mala.Parameters()
        parameters.targets.ldos_gridoffset_ev = -5
        parameters.targets.ldos_gridsize = 11
        parameters.targets.ldos_gridspacing_ev = 2.5

        # initialize and add snapshots to workflow
        ldos_aligner = mala.LDOSAligner(parameters)
        ldos_aligner.clear_data()
        ldos_aligner.add_snapshot("Be_snapshot0.out.npy", data_path_be)
        ldos_aligner.add_snapshot("Be_snapshot1.out.npy", data_path_be)
        ldos_aligner.add_snapshot("Be_snapshot2.out.npy", data_path_be)

        # align and cut the snapshots from the left and right-hand sides
        ldos_aligner.align_ldos_to_ref(
            left_truncate=True, right_truncate_value=11, number_of_electrons=4
        )

        try:
            import openpmd_api

            use_openpmd = True
        except ImportError:
            use_openpmd = False

        if use_openpmd:
            # initialize and add snapshots to workflow
            ldos_aligner = mala.LDOSAligner(parameters)
            ldos_aligner.clear_data()
            ldos_aligner.add_snapshot(
                "Be_snapshot0.out.h5", data_path_be, snapshot_type="openpmd"
            )
            ldos_aligner.add_snapshot(
                "Be_snapshot1.out.h5", data_path_be, snapshot_type="openpmd"
            )
            ldos_aligner.add_snapshot(
                "Be_snapshot2.out.h5", data_path_be, snapshot_type="openpmd"
            )

            # align and cut the snapshots from the left and right-hand sides
            ldos_aligner.align_ldos_to_ref(
                left_truncate=True,
                right_truncate_value=11,
                number_of_electrons=4,
            )

            parameters = mala.Parameters()
            data_handler = mala.DataHandler(parameters)
            for i in range(1, 4):
                data_openpmd = (
                    data_handler.target_calculator.read_from_openpmd_file(
                        f"{data_path_be}/aligned/Be_snapshot0.out.h5"
                    )
                )
                data_numpy = (
                    data_handler.target_calculator.read_from_numpy_file(
                        f"{data_path_be}/aligned/Be_snapshot0.out.npy"
                    )
                )
                if not np.allclose(data_numpy, data_openpmd):
                    raise Exception("Inconsistency in snapshot", i)

    def test_ldos_splitting_multiple_elements(self):
        """
        Test that the LDOS splitting works both on LDOS and DOS level.

        We compute the band energy with splitted and unsplitted DOS and
        compare both to splitted LDOS band energy.
        """
        params = mala.Parameters()

        params.targets.ldos_gridsize = [12, 13, 14, 28]
        params.targets.ldos_gridspacing_ev = [0.5, 0.5, 0.5, 0.5]
        params.targets.ldos_gridoffset_ev = [-19, -10.5, -4.5, 3.5]
        params.targets.pseudopotential_path = "."

        dos_calculator = mala.DOS(params)
        dos_calculator.read_additional_calculation_data(
            os.path.join(data_path_bao, "BaO_snapshot0.out")
        )
        dos_calculator.read_from_qe_out(
            os.path.join(data_path_bao, "BaO_snapshot0.out"),
            smearing_factor=[2, 2, 2, 2],
        )

        params2 = mala.Parameters()
        params2.targets.ldos_gridsize = 73
        params2.targets.ldos_gridspacing_ev = 0.5
        params2.targets.ldos_gridoffset_ev = -19
        params2.targets.pseudopotential_path = "."
        dos_calculator_unsplitted = mala.DOS(params2)
        dos_calculator_unsplitted.read_additional_calculation_data(
            os.path.join(data_path_bao, "BaO_snapshot0.out")
        )

        dos_calculator_unsplitted.read_from_qe_out(
            os.path.join(data_path_bao, "BaO_snapshot0.out"),
            smearing_factor=2,
        )

        params3 = mala.Parameters()
        params3.targets.ldos_gridsize = [12, 13, 14, 28]
        params3.targets.ldos_gridspacing_ev = [0.5, 0.5, 0.5, 0.5]
        params3.targets.ldos_gridoffset_ev = [-19, -10.5, -4.5, 3.5]

        ldos_calculator = mala.LDOS.from_numpy_file(
            params3, os.path.join(data_path_bao, "BaO_snapshot0.out.npy")
        )
        ldos_calculator.read_additional_calculation_data(
            os.path.join(data_path_bao, "BaO_snapshot0.info.json")
        )
        assert np.isclose(
            dos_calculator.band_energy,
            ldos_calculator.band_energy,
            atol=accuracy_strict,
        )
        assert np.isclose(
            dos_calculator.band_energy,
            dos_calculator_unsplitted.band_energy,
            atol=accuracy_band_energy,
        )
