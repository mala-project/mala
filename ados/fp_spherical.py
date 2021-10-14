import sys
import numpy as np
import argparse
import os
import time
import scipy
from pprint import pprint
from scipy.spatial.distance import pdist
from pymatgen.io.lammps.data import LammpsData
from pymatgen.transformations.standard_transformations import RotationTransformation
from mesh.mesh import *


class ConfigurationFingerprint(object):
    """
    Maps configuration file to icosahedral spherical mesh signals
    Output dim: [N x R x V]
    """

    def __init__(self, r_atom, r_dim, l, map_type, scaling_type):
        self.r_atom = r_atom
        self.r_dim = r_dim
        self.mesh_lvl = l
        self.map_type = map_type
        self.scaling_type = scaling_type
        self.mesh = icosphere(self.mesh_lvl) # TODO: this should be generated once and saved to file
        self.mesh_knn = scipy.spatial.cKDTree(self.mesh.vertices)
        
        self.scaling_fn = None
        if scaling_type == "none":
            # identity
            self.scaling_fn = lambda x, r: x
        elif scaling_type == "inverse":
            self.scaling_fn = lambda x, r: x/r
        elif scaling_type == "inverse_sq":
            self.scaling_fn = lambda x, r: x/(r**2)

        self.smoothing_fn = None
        if map_type == "linear":
            self.mapping = self.map_linear
        elif map_type == "sqrt":
            self.mapping = self.map_sqrt
        elif map_type == "log":
            self.mapping = self.map_log
            

    def __call__(self, pos, neighbor_pos, neighbor_dists, rotate=False):
        features = self.mapping(neighbor_pos, neighbor_dists, self.r_dim, self.r_atom, r_min=2)
        #print("time to map data: {}".format(time.perf_counter() - tic))
        return features


    def map_linear(self, environments, neighbor_dists, R, r_max, r_min=0):
        """
        Output: [R, V, 1]
        """

        r_range = r_max - r_min
        scale = np.array([i+1 for i in range(R)])
        #print("scale: {}".format(scale))
        s_max = scale[-1]
        rcuts = (scale/s_max)*r_range
        #print("rcuts: {}".format(rcuts))
        
        spherical_signal = np.zeros((R, self.mesh.vertices.shape[0]))
        pts = environments
        pts_unit = normalize_to_radius(pts, 1)
        _, mesh_idx = self.mesh_knn.query(pts_unit, k=1, n_jobs=1) # nearest vertex
        p_dists = neighbor_dists
        p_dists_scaled = p_dists - r_min
        r_idx = (np.ceil(p_dists_scaled*s_max / r_range)-1).astype(int)

        Z = self.scaling_fn(np.ones_like(mesh_idx), p_dists)
        for i, (r_idx, v_idx) in enumerate(zip(r_idx, mesh_idx)):
            spherical_signal[r_idx, v_idx] += Z[i]

        spherical_signal = spherical_signal[:, :, np.newaxis]

        return spherical_signal


def spherical_neighbors(config_structure, r, rotate=False):
    """
    Finds coordinates and distances of neighbors, centered at reference point

    :param r: radius cut-off
    """
    supercell = config_structure
    """
    if rotate:
        axis, angle = random_axis_angle()
        #print("axis: {}, angle: {}".format(axis, angle))
        rot = RotationTransformation(axis, angle, angle_in_radians=True)
        supercell = rot.apply_transformation(supercell)
    """

    atom_sites = supercell.sites
    s0 = atom_sites[0]

    atom_coords = np.array([s.coords for s in atom_sites])
    center_idx, point_idx, offsets, dists = supercell.get_neighbor_list(r, atom_sites)
    neighbor_coords = atom_coords[point_idx] + offsets.dot(s0.lattice.matrix)
    
    neighbor_counts = np.zeros(len(atom_sites), dtype=int)
    for c_idx in center_idx:
        neighbor_counts[c_idx] += 1

    atom_neighbors = []
    atom_neighbor_dists = []
    neighbor_offsets = np.zeros(len(atom_sites)+1, dtype=int)
    for i, count in enumerate(neighbor_counts):
        neighbor_offsets[i+1] = neighbor_offsets[i] + count
    
    for i in range(len(neighbor_counts)):
        off1, off2 = neighbor_offsets[i], neighbor_offsets[i+1]
        atom_neighbors.append(neighbor_coords[off1:off2])
        atom_neighbor_dists.append(dists[off1:off2])


    neighb_coords_centered = []
    neighb_dists_scaled = []
    for (coords, neighb_coords, dists) in zip(atom_coords, atom_neighbors, atom_neighbor_dists):
        if len(neighb_coords):
            neighbors_c = neighb_coords-coords
            neighb_coords_centered.append(neighbors_c)

    atom_env_data = list(zip(atom_coords, neighb_coords_centered, atom_neighbor_dists))
    return atom_env_data


def random_axis_angle():
    """
    Does not necessarily correspond to uniformly random rotation because
    angle should be *non-uniformly* sampled
    """
    v = np.random.normal(size=3)
    v_unit = v/np.linalg.norm(v)
    angle = np.random.uniform(0, 2*np.pi)
    return v_unit, angle

        
def rbf(s, gamma=1):
    """
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    Input: scalar
    """
    return np.exp(-gamma*(s**2))


def normalize_to_radius(points, radius):
    '''
    Reproject points to specified radius(es)
    - points: array
    - radius: scalar or array
    '''
    scalar = (points**2).sum(axis=-1, keepdims=True)**.5
    unit = points / scalar
    offset = radius - scalar
    points_new = points + unit*offset
    return points_new


def rand_rotate_points(X, R):
    """
    Rotates 3D points, returns copy
    Input dims: [..., 3]
    """
    # apply matrix-vector product
    X_r = np.einsum('...ij,...j', R, X)
    return X_r


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    
    # https://github.com/jonas-koehler/s2cnn.git
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


if __name__ == '__main__':

    """
    parser = argparse.ArgumentParser(description='Generate atomic environment fingerprints')
    parser.add_argument('-r', help='Neighborhood radius for molecular environment', type=float, default=4.67637)
    parser.add_argument('-R', type=int, default=32, help='Number of radial levels')
    parser.add_argument('-s', '--snapshot', type=int, default=0, help='Snapshot id')
    parser.add_argument('-l', '--level', help='Level of mesh refinement', type=int, dest='l', default=3)
    parser.add_argument("-m", default="linear", choices=["linear", "sqrt", "log"], help="Data mapping type")
    parser.add_argument("-scaling", default="inverse", choices=["none", "inverse", "inverse_sq"], help="Distance scaling type")
    parser.add_argument("--rotate", action='store_true', help="Augment training with rotations")

    args = parser.parse_args()
    fp = ConfigurationFingerprint(args.r, args.R, args.l, args.m, args.scaling)

    data_dir = os.environ["ALUMINUM_DIR"]
    snapshot = "Al.scf.pw.snapshot{}.lammps".format(args.snapshot)
    fpath = os.path.join(data_dir, "933K/2.699gcc", snapshot)
    print(fpath)
    config_data = LammpsData.from_file(fpath, atom_style="atomic")

    N_ATOMS = len(config_data.structure.sites)
    print(N_ATOMS)
    features,_ = fp(config_data, N_ATOMS)
    print(features.shape)
    v_nnz_idx = np.where(np.sum(features[0, 0], axis=-1) > 0)
    pprint(features[0, 0, v_nnz_idx]) # descriptor of 1 atom
    """

