import os

import mala
from mala.datahandling.data_repo import data_path

"""
Shows how MALA can be used to optimize descriptor
parameters based on the ACSD analysis (see hyperparameter paper in the
documentation for mathematical details).
"""

####################
# 1. DETAILS OF THE ACSD ANALYSIS
# Define how many points should be used for the ACSD analysis
# and which values should be used for the bispectrum hyperparameters.
####################
parameters = mala.Parameters()

# Specify the details of the ACSD analysis.
parameters.descriptors.acsd_points = 100
hyperoptimizer = mala.ACSDAnalyzer(parameters)
hyperoptimizer.add_hyperparameter("bispectrum_twojmax", [2, 4])
hyperoptimizer.add_hyperparameter("bispectrum_cutoff", [1.0, 2.0])

####################
# 2. DATA
# When adding data for the ACSD analysis, add preprocessed LDOS data for
# and a calculation output for the descriptor calculation.
####################
hyperoptimizer.add_snapshot(
    "espresso-out",
    os.path.join(data_path, "Be_snapshot1.out"),
    "numpy",
    os.path.join(data_path, "Be_snapshot1.out.npy"),
    target_units="1/(Ry*Bohr^3)",
)
hyperoptimizer.add_snapshot(
    "espresso-out",
    os.path.join(data_path, "Be_snapshot2.out"),
    "numpy",
    os.path.join(data_path, "Be_snapshot2.out.npy"),
    target_units="1/(Ry*Bohr^3)",
)

# If you plan to plot the results (recommended for exploratory searches),
# the optimizer can return the necessary quantities to plot.
hyperoptimizer.perform_study(return_plotting=False)
hyperoptimizer.set_optimal_parameters()
