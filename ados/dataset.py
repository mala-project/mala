import sys
import glob
import os
import numpy as np
import torch
import torch.utils.data
import argparse
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.structure import Structure
from .fp_spherical import ConfigurationFingerprint, spherical_neighbors
import ados.DFT_calculators as DFT_calculators


def load_snapshots_list(filename):
    ss = []
    with open(filename, "r") as f:
        for line in f:
            temp, density, ss_id, phase = line.split()
            ss.append((temp, density, ss_id, phase))
    return ss


class AtomicConfigurations(torch.utils.data.Dataset):


    def __init__(self, root, snapshots, rcut, data_transform=None, target_transform=None, rotate=False):
        self.root = os.path.expanduser(root)
        self.snapshots = snapshots
        self.rcut = rcut
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.rotate = rotate
        self.splits = None

        self.ref_files, dft_files = self.get_files()

        self.atom_env_data = []
        self.ref_data = []
        # The index_config_map is used to look up the snapshot/config that a sample belongs to
        self.index_config_map = []

        self.dft_results = []
        for i, dft_file in enumerate(dft_files):
            dft_result = DFT_calculators.DFT_results(dft_file)
            self.dft_results.append(dft_result)
            dft_pos = dft_result.positions
            n_atoms = len(dft_pos)
            config_species = ["Al" for _ in range(len(dft_pos))]
            config_structure = Structure(dft_result.cell, config_species, dft_pos, 0, coords_are_cartesian=True)
            local_env_data = spherical_neighbors(config_structure, self.rcut)
            self.atom_env_data.extend(local_env_data)
            self.index_config_map.extend([self.snapshots[i]]*n_atoms)

        for ref_file in self.ref_files:
            ados_data = np.load(ref_file)
            for ados in ados_data:
                self.ref_data.append(torch.FloatTensor(ados))


    def get_files(self):
        data_fpaths = []
        dft_fpaths = []
        for temp, density, ss_id, _ in self.snapshots:
            sshot_dir = os.path.join(self.root, f"{temp}K/{density}gcc")
            data_fpath = os.path.join(sshot_dir, f"Al_ados_250elvls_sigma1.3_snapshot{ss_id}.npy")
            data_fpaths.append(data_fpath)
            dft_fpath = os.path.join(sshot_dir, f"QE_Al.scf.pw.snapshot{ss_id}.out")
            dft_fpaths.append(dft_fpath)

        return data_fpaths, dft_fpaths


    def __getitem__(self, index):
        pos, neighbor_pos, neighbor_dists = self.atom_env_data[index]
        y = self.ref_data[index]
        x = self.data_transform(pos, neighbor_pos, neighbor_dists, self.rotate)
        x = torch.FloatTensor(x)
        # Return the index so it can be used to look up the config/snapshot using
        # self.index_config_map
        items = [x, y, index]
        return items


    def __len__(self):
        return len(self.atom_env_data)



if __name__ == "__main__":

    data_dir = os.path.join(os.environ["DATA_DIR"], "aluminum")
    r_cutoff = 5
    R=8
    l=3
    temp = 933
    gcc = 2.699
    transform = ConfigurationFingerprint(r_cutoff, R, l, 'linear', 'inverse')
    mode = "train"
    ados_data = AtomicConfigurations(data_dir, mode, temp, r_cutoff, data_transform=transform)
    x, y, config_id = ados_data[-1]
    #print(y)

    """
    x = x.numpy()
    v_nnz_idx = np.where(np.sum(x[0], axis=-1) > 0)
    print(v_nnz_idx)
    print(len(v_nnz_idx))
    nnz = x[0, v_nnz_idx]
    pprint(nnz[0])
    """
