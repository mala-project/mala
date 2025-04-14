import os

import numpy

import mala

from mala.datahandling.data_repo import data_path

"""
Shows how to align the energy spaces of different LDOS vectors to a reference.
This is useful when the band energy spectrum starts at different values, e.g.
when MALA is trained for snapshots of different mass densities.

Note that this example is only a proof-of-principle, because the alignment
algorithm has no effect on the Be test snapshots (apart from truncation).
"""


parameters = mala.Parameters()
parameters.targets.ldos_gridoffset_ev = -5
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5

# initialize and add snapshots to workflow
ldos_aligner = mala.LDOSAligner(parameters)
ldos_aligner.clear_data()
ldos_aligner.add_snapshot("Be_snapshot0.out.npy", data_path)
ldos_aligner.add_snapshot("Be_snapshot1.out.npy", data_path)
ldos_aligner.add_snapshot("Be_snapshot2.out.npy", data_path)

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
    ldos_aligner.add_snapshot("Be_snapshot0.out.h5",
                              data_path, snapshot_type='openpmd')
    ldos_aligner.add_snapshot("Be_snapshot1.out.h5",
                              data_path, snapshot_type='openpmd')
    ldos_aligner.add_snapshot("Be_snapshot2.out.h5",
                              data_path, snapshot_type='openpmd')

    # align and cut the snapshots from the left and right-hand sides
    ldos_aligner.align_ldos_to_ref(
        left_truncate=True, right_truncate_value=11, number_of_electrons=4
    )

    parameters = mala.Parameters()
    data_handler = mala.DataHandler(parameters)
    for i in range(1, 4):
        data_openpmd = data_handler.target_calculator.read_from_openpmd_file(
            f"{data_path}/aligned/Be_snapshot0.out.h5")
        data_numpy = data_handler.target_calculator.read_from_numpy_file(
            f"{data_path}/aligned/Be_snapshot0.out.npy")
        if not numpy.allclose(data_numpy, data_openpmd):
            raise Exception("Inconsistency in snapshot", i)
        else:
            print("All close.")
