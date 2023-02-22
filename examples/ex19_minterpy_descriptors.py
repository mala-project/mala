import os
import mala
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

outfile = os.path.join(data_path, "Be_snapshot0.out")

params = mala.Parameters()
params.descriptors.atomic_density_cutoff = 4.0
params.descriptors.descriptors_contain_xyz = True
params.descriptors.minterpy_cutoff_cube_size = 4.0
params.descriptors.minterpy_polynomial_degree = 4
params.descriptors.minterpy_lp_norm = 2

descriptor_calculator = mala.MinterpyDescriptors(params)
data = descriptor_calculator.calculate_from_qe_out(outfile)[0]

print(data[0, 0, :, :])

