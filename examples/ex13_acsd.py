import os

import mala
import numpy as np
from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex13_calculate_acsd.py: Shows how MALA can be used to optimize descriptor
parameters based on the ACSD analysis (see hyperparameter paper in the 
documentation for mathematical details).
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()

# Specify the details of the ACSD analysis.
test_parameters.descriptors.descriptors_contain_xyz = True
test_parameters.descriptors.acsd_points = 100

# Prepare the analysis by adding those descriptor parameters that should be
# optimized.
hyperoptimizer = mala.ACSDAnalyzer(test_parameters)
hyperoptimizer.add_hyperparameter("bispectrum_twojmax", [2, 4])
hyperoptimizer.add_hyperparameter("bispectrum_cutoff", [1.0, 2.0])

# Add raw snapshots to the hyperoptimizer. For the targets, numpy files are
# okay as well.
hyperoptimizer.add_snapshot("espresso-out", os.path.join(data_path, "Be_snapshot1.out"),
                            "numpy", os.path.join(data_path, "Be_snapshot1.out.npy"),
                            target_units="1/(Ry*Bohr^3)")
hyperoptimizer.add_snapshot("espresso-out", os.path.join(data_path, "Be_snapshot2.out"),
                            "numpy", os.path.join(data_path, "Be_snapshot2.out.npy"),
                            target_units="1/(Ry*Bohr^3)")

# If you plan to plot the results (recommended for exploratory searches),
# the optimizer can return the necessary quantities to plot.
plotting = hyperoptimizer.perform_study(return_plotting=True)
hyperoptimizer.set_optimal_parameters()

do_plot = False
if do_plot:
    # With a script like this, the ACSD can be plotted.
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    ax.set_xticks(np.arange(len(plotting[1]["twojmax"])))
    ax.set_xticklabels(plotting[1]["twojmax"])
    ax.set_yticks(np.arange(len(plotting[1]["cutoff"])))
    ax.set_yticklabels(plotting[1]["cutoff"])

    plt.imshow(plotting[0], cmap='hot', interpolation='nearest')
    plt.show()

