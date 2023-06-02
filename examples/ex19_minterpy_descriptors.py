import os
import mala
from mala import printout

import numpy as np
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
ex19_minterpy_descriptors.py: To calculate the input data
using the minterpy descriptors.

REQUIRES LAMMPS and MINTERPY (https://github.com/casus/minterpy).
"""

outfile = os.path.join(data_path, "Be_snapshot0.out")

params = mala.Parameters()
params.descriptors.atomic_density_cutoff = 4.0
params.descriptors.descriptors_contain_xyz = True
# cutoff to determine the size of local cubes around each grid 
params.descriptors.minterpy_cutoff_cube_size = 4.0
# polynomial degree to interpolate atomic densities on grids
params.descriptors.minterpy_polynomial_degree = 4
params.descriptors.minterpy_lp_norm = 2

descriptor_calculator = mala.MinterpyDescriptors(params)
data = descriptor_calculator.calculate_from_qe_out(outfile)[0]

printout(data.shape)
np.save('Be_snapshot0.in.npy', data)
