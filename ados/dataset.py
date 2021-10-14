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


class AtomicConfigurations(torch.utils.data.Dataset):


    def __init__(self, root, mode, temp, rcut, data_transform=None, target_transform=None, rotate=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.temp = temp
        self.rcut = rcut
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.rotate = rotate
        self.splits = None
        if temp == 298:
            self.splits = {
                "train": [*range(0, 8)],
                "validation": [8],
                "test": [9]
            }
        elif temp == 933:
            self.splits = {
                "train": [*range(0, 6)] + [*range(10, 16)],
                "validation": (6, 16),
                "test": [*range(7, 10)] + [*range(17, 20)]
            }

        if temp not in (298, 933):
            raise Exception("Temp {}K dataset not implemented".format(temp))
        if mode not in ["train", "validation", "test"]:
            raise Exception("Invalid dataset mode {}".format(mode))

        self.config_ids, self.ref_files, dft_files = self.get_files()

        self.atom_env_data = []
        self.ref_data = []
        self.atom_config_ids = []

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
            config_id = self.config_ids[i]
            self.atom_config_ids.extend(np.full(n_atoms, config_id))

        for ref_file in self.ref_files:
            ados_data = np.load(ref_file)
            for ados in ados_data:
                self.ref_data.append(torch.FloatTensor(ados))


    def get_files(self):
        data_fpaths = []
        dft_fpaths = []
        sshot_ids = self.splits[self.mode]
        sshot_dir = os.path.join(self.root, "{}K/2.699gcc".format(self.temp))
        for sshot_id in sshot_ids:
            sshot_fpath = os.path.join(sshot_dir, "Al.scf.pw.snapshot{}.lammps".format(sshot_id))
            data_fpath = os.path.join(sshot_dir, "Al_ados_250elvls_sigma1.3_snapshot{}.npy".format(sshot_id))
            data_fpaths.append(data_fpath)
            dft_fpath = os.path.join(sshot_dir, "QE_Al.scf.pw.snapshot{}.out".format(sshot_id))
            dft_fpaths.append(dft_fpath)

        return sshot_ids, data_fpaths, dft_fpaths


    def __getitem__(self, index):
        pos, neighbor_pos, neighbor_dists = self.atom_env_data[index]
        config_id = self.atom_config_ids[index]
        y = self.ref_data[index]
        x = self.data_transform(pos, neighbor_pos, neighbor_dists, self.rotate)
        x = torch.FloatTensor(x)
        items = [x, y, config_id]

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
