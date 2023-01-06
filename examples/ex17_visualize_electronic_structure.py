import os

from ase.io import read
import mala
import numpy as np

from mala.datahandling.data_repo import data_repo_path
density_path = os.path.join(os.path.join(data_repo_path, "Be2"),
                          "Be_dens.npy")
atoms_path = os.path.join(os.path.join(data_repo_path, "Be2"),
                          "Be.pw.scf.out")
"""
ex17_visualize_electronic_structure.py : Show how MALA can output
prediction results in a file format that can be processed by electronic 
structure readers.
"""


# Our starting point is a numpy array containing the density.
# This would be the way MALA gives us back network predictions.
# To further model the MALA behavior, we'll reshape it into 1D.
# Regular MALA inference will return a 1D density.
density = np.load(density_path)
grid_dimensions = [density.shape[0], density.shape[1], density.shape[2]]
density = np.reshape(density, [np.prod(grid_dimensions), 1])

# In the inference case the command that reads grid information is
# automatically done in the prediction routine.
# Here we create a density calculator and write out the density.
atoms = read(atoms_path)
params = mala.Parameters()
density_calculator = mala.Density(params)
density_calculator.read_additional_calculation_data([atoms, grid_dimensions])

# The resulting cube file can be opened by e.g. VMD, VESTA or scenery.
density_calculator.write_to_cube("Be_dens.cube", density, atoms, grid_dimensions)

# Writing of a file for OpenPMD requires reshaping.
# OpenPMD will become the interface for Unity visualization.
# density3D = np.reshape(density, grid_dimensions + [1])
# density_calculator.write_to_openpmd_file("Be_dens.h5", density3D)
