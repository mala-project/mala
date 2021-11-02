import mala
from mala.common.parallelizer import get_rank
import numpy as np

"""
ex15: Work in progress
"""

params = mala.Parameters()
params.use_mpi = True
snap = mala.SNAP(params)
snap_descriptors = snap.calculate_from_qe_out("Be_snapshot1.out",
                                              "/home/fiedlerl/data/"
                                              "mala_data_repo/Be2/"
                                              "training_data")
snap_descriptors = snap.gather_descriptors(snap_descriptors)
# Comparison
total_error = []
if get_rank() == 0:
    comparison_sd = np.load("/home/fiedlerl/data/mala_data_repo/Be2/"
                            "training_data/Be_snapshot1.in.npy")
    for x in range(0,18):
        for y in range(0, 18):
            for z in range(0, 27):
                for f in range(0, 94):
                    if comparison_sd[x, y, z, f] != 0.0:
                        total_error.append(np.abs((snap_descriptors[x, y, z, f] -
                                                  comparison_sd[x, y, z, f]) /
                                                  comparison_sd[x, y, z, f]) * 100)
    print(np.mean(total_error))

