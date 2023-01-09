import os
import mala
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

outfile = os.path.join(data_path, "Be.pw.scf.out")

params = mala.Parameters()
params.descriptors.atomic_density_cutoff = 4.0
params.descriptors.descriptors_contain_xyz = True
params.descriptors.minterpy_point_list = [
    [0.25, 0.25, 0.25],
    [0.5, 0.5, 0.5],
    [0.75, 0.75, 0.75]
]

# Set the size of the local cubes.
params.descriptors.minterpy_cutoff_cube_size = 0.0

descriptor_calculator = mala.MinterpyDescriptors(params)
data = descriptor_calculator.calculate_from_qe_out(outfile)[0]

print(data[0, 0, :, :])

