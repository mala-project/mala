import os

import numpy as np

import mala
from mala.datahandling.data_repo import data_path_be


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
