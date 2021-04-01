import numpy as np
import torch
import utils
import e3nn
from e3nn.rs import dim, mul_dim, map_mul_to_Rs
import e3nn.o3 as o3
from e3nn.spherical_harmonics import SphericalHarmonicsFindPeaks


# -

__authors__  = "Tess E. Smidt, Mario Geiger"

torch.set_default_dtype(torch.float64)

def direct_sum(*matrices):
    # Slight modification of se3cnn.SO3.direct_sum
    """
    Direct sum of matrices, put them in the diagonal
    """

    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(*total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x 
        i += m
        j += n
    return out 


def adjusted_projection(vectors, L_max, sum_points=True, radius=True):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.]

    if radius:
        radii = radii[radii > 0.]
    else:
        radii = torch.ones_like(radii[radii > 0.])

    angles = e3nn.o3.xyz_to_angles(vectors)
    coeff = e3nn.o3.spherical_harmonics_dirac(L_max, *angles)
    coeff *= radii.unsqueeze(-2)
    
    A = torch.einsum("ia,ib->ab", (e3nn.o3.spherical_harmonics(list(range(L_max + 1)), *angles), coeff))
    try:
        coeff *= torch.lstsq(radii, A).solution.view(-1)
    except:
        coeff *= torch.gels(radii, A).solution.view(-1)
    return coeff.sum(-1) if sum_points else coeff


class SphericalTensor():
    def __init__(self, signal, Rs):
        self.signal = signal
        self.Rs = Rs
    
    @classmethod
    def from_geometry(cls, vectors, L_max, sum_points=True, radius=True):
        Rs = [(1, L) for L in range(L_max + 1)]
        signal = adjusted_projection(vectors, L_max, sum_points=sum_points, radius=radius)
        return cls(signal, Rs)

    @classmethod
    def from_geometry_with_radial(cls, radial_model, vectors, L_max, sum_points=True):
        radial_functions = radial_model(vectors)  # [N, R]
        N, R = radial_functions.shape
        Rs = [(R, L) for L in range(L_max + 1)]
        signal = adjusted_projection(vectors, L_max, sum_points=False,
                                     radius=False)  # [channels, N]
        sphten = torch.einsum('nr,cn->cr', radial_functions,
                              signal).reshape(-1)
        new_cls = cls(signal, Rs)
        new_cls.radial_model = radial_model
        return new_cls

    @classmethod
    def from_decorated_geometry(cls, vectors, features, L_max,
                                features_Rs=None, sum_points=True,
                                radius=True):
        Rs = [(1, L) for L in range(L_max + 1)]
        # [Rs, points]
        geo_signal = adjusted_projection(coords, L_max, sum_points=False, radius=radius)
        # Keep Rs index for geometry and features
        new_signal = torch.einsum('gp,pf->fg', (geo_signal, features))
        new_signal = cls(new_signal, Rs)
        new_signal.feature_Rs = features_Rs
        return new_signal

    def sph_norm(self):
        Rs = self.Rs
        signal = self.signal
        n_mul = sum([mul for mul, L in Rs])
        # Keep shape after Rs the same
        norms = torch.zeros(n_mul, *signal.shape[1:])
        sig_index = 0
        norm_index = 0
        for mul, L in Rs:
            for m in range(mul):
                norms[norm_index] = signal[sig_index: sig_index +
                                           (2 * L + 1)].norm(2, 0)
                norm_index += 1
                sig_index += 2 * L + 1
        return norms

    def signal_on_sphere(self, which_mul=None, n=100, radius=True):
        n_mul = sum([mul for mul, L in self.Rs])
        if which_mul:
            if len(which_mul) != n_mul:
                raise ValueError("which_mul and number of multiplicities is " +
                                 "not equal.")
        else:
            which_mul = [1 for i in range(n_mul)]

        # Need to handle if signal is featurized
        x, y, z = (None, None, None)
        Ys = []
        for mul, L in self.Rs:
            # Using cache-able function
            x, y, z, Y = utils.spherical_harmonics_on_grid(L, n)
            Ys += [Y] * mul

        f = self.signal.unsqueeze(1).unsqueeze(2) * torch.cat(Ys, dim=0)
        f = f.sum(0)
        return x, y, z, f

    def plot(self, which_mul=None, n=100, radius=True, center=None, relu=True):
        """
        surface = self.plot()
        fig = go.Figure(data=[surface])
        fig.show()
        """
        import plotly.graph_objs as go

        x, y, z, f = self.signal_on_sphere(which_mul, n, radius)
        f = f.relu() if relu else f

        if radius:
            r = f.abs()
            x = x * r
            y = y * r
            z = z * r

        if center is not None:
            x = x + center[0]
            y = y + center[1]
            z = z + center[2]

        return go.Surface(x=x.numpy(), y=y.numpy(), z=z.numpy(), surfacecolor=f.numpy())

    def plot_with_radial(self, box_length, n=100, center=None,
                         sh=o3.spherical_harmonics_xyz):
        muls, Ls = zip(*Rs)
        # We assume radial functions are repeated across L's
        assert len(set(muls)) == 1
        num_L = len(Rs)
        new_radial = lambda x: x.repeat(1, num_L) # Repeat along filter dim
        r, f = plot_data_on_grid(box_length, new_radial, self.Rs, sh=sh, n=n)
        # Multiply coefficients
        return r, torch.einsum('xd,d->x', f, self.signal)

    def wigner_D_on_grid(self, n):
        try:
            return getattr(self, "wigner_D_grid_{}".format(n))
        except:
            blocks = [utils.wigner_D_on_grid(L, n)
                      for mul, L in self.Rs for m in range(mul)]
            wigner_D = direct_sum(*blocks)
            setattr(self, "wigner_D_grid_{}".format(n), wigner_D)
            return getattr(self, "wigner_D_grid_{}".format(n))

    def cross_correlation(self, other, n, normalize=True):
        if self.Rs != other.Rs:
            raise ValueError("Rs must match")
        wigner_D = self.wigner_D_on_grid(n)
        normalize_by = (self.signal.norm(2, 0) * other.signal.norm(2, 0))
        cross_corr =  torch.einsum(
            'abcji,j,i->abc', (wigner_D, self.signal, other.signal)
        )
        return cross_corr / normalize_by if normalize else cross_corr

    def find_peaks(self, which_mul=None, n=100, min_radius=0.1,
                   percentage=False, absolute_min=0.1, radius=True):
        import scipy.signal

        if not hasattr(self, 'peak_finder') or self.peak_finder.n != n:
            L_max = max(L for mult, L in self.Rs)
            self.peak_finder = SphericalHarmonicsFindPeaks(n, L_max)

        peaks, radius = self.peak_finder.forward(self.signal) 

        if percentage:
            self.used_radius = max((min_radius * torch.max(radius)),
                                   absolute_min)
            keep_indices = (radius > max((min_radius * torch.max(radius)),
                                         absolute_min))
        else:
            self.used_radius = min_radius
            keep_indices = (radius > min_radius)
        return peaks[keep_indices] * radius[keep_indices].unsqueeze(-1)

    def __add__(self, other):
        if self.Rs == other.Rs:
            from copy import deepcopy
            return SphericalTensor(self.signal + other.signal,
                                   deepcopy(self.Rs))

    def __mul__(self, other):
        # Dot product if Rs of both objects match
        # Add check for feature_Rs.
        if self.Rs == other.Rs:
            dot = (self.signal * other.signal).sum(-1)
            dot /= (self.signal.norm(2, 0) * other.signal.norm(2, 0))
            return dot

    def __matmul__(self, other):
        # Tensor product
        # Assume first index is Rs
        # Better handle mismatch of features indices
        Rs_out, C = e3nn.rs.tensor_product(self.Rs, other.Rs)
        Rs_out = [(mult, L) for mult, L, parity in Rs_out]
        new_signal = torch.einsum('ijk,i...,j...->k...', 
                                  (C.permute(1,2,0), self.signal, other.signal))
        return SphericalTensor(new_signal, Rs_out)

    def __rmatmul__(self, other):
        # Tensor product
        return self.__matmul__(self, other)


class VisualizeKernel():
    def __init__(self, Kernel):
        self.kernel = Kernel

    def Ys_on_sphere(self, n=100):
        x, y, z = (None, None, None)
        Ys = []
        L_to_index = {}
        start = 0
        for L in self.kernel.set_of_l_filters:
            # Using cache-able function
            x, y, z, Y = utils.spherical_harmonics_on_grid(L, n)
            Ys += [Y]
            L_to_index[L] = [start, start + 2 * L + 1]
            start += 2 * L + 1

        return x, y, z, torch.cat(Ys, dim=0), L_to_index

    def plot_data(self, max_radius, n=50, nr=10, min_radius=0.1):
        """
        surface = self.plot()
        fig = go.Figure(data=[surface])
        fig.show()
        """
        import e3nn
        from e3nn.rs import dim

        Rs_filter = [(1, L) for L in self.kernel.list_of_l_filters]

        x, y, z, Ys, L_to_index = self.Ys_on_sphere(n)

        r_values = torch.linspace(min_radius, max_radius, nr)
        R = self.kernel.R(r_values).detach()  # [r_values, n_filters]

        R_helper = torch.zeros(R.shape[-1], dim(Rs_filter))
        start = 0
        Ys_indices = []
        for i, (mul, L) in enumerate(Rs_filter):
            R_helper[i, start: start + 2 * L + 1] = 1.
            start += 2 * L + 1
            Ys_indices += list(range(L_to_index[L][0], L_to_index[L][1]))

        full_Ys = Ys[Ys_indices]  # [theta_values, dim(Rs_filter)]]  
        full_Ys = full_Ys.reshape(full_Ys.shape[0], -1)
        all_x = (r_values.unsqueeze(-1) * x.flatten().unsqueeze(0)).flatten()
        all_y = (r_values.unsqueeze(-1) * y.flatten().unsqueeze(0)).flatten()
        all_z = (r_values.unsqueeze(-1) * z.flatten().unsqueeze(0)).flatten()
        all_f = torch.einsum('rn,nd,da->rad', R, R_helper, full_Ys)
        all_f = all_f.reshape(-1, all_f.shape[-1])

        return all_x, all_y, all_z, all_f

    def plot_data_on_grid(self, box_length, n=30):
        import e3nn
        from e3nn.rs import dim

        Rs_filter = [(1, L) for L in self.kernel.list_of_l_filters]

        L_to_index = {}
        start = 0
        for L in self.kernel.set_of_l_filters:
            L_to_index[L] = [start, start + 2 * L + 1]
            start += 2 * L + 1

        r = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j].reshape(3, -1)
        r = r.transpose(1, 0)
        r *= box_length / 2.
        r = torch.from_numpy(r)
        Ys = self.kernel.sh(self.kernel.set_of_l_filters, r)
        R = self.kernel.R(r.norm(2, -1)).detach()  # [r_values, n_filters]

        R_helper = torch.zeros(R.shape[-1], dim(Rs_filter))
        start = 0
        Ys_indices = []
        for i, (mul, L) in enumerate(Rs_filter):
            R_helper[i, start: start + 2 * L + 1] = 1.
            start += 2 * L + 1
            Ys_indices += list(range(L_to_index[L][0], L_to_index[L][1]))

        full_Ys = Ys[Ys_indices]  # [values, dim(Rs_filter)]]  
        full_Ys = full_Ys.reshape(full_Ys.shape[0], -1)
        all_f = torch.einsum('xn,nd,dx->xd', R, R_helper, full_Ys)
        all_f = all_f.reshape(-1, all_f.shape[-1])
        return r, all_f


def plot_data_on_grid(box_length, radial, Rs, sh=o3.spherical_harmonics_xyz,
                      n=30, center=None):
    L_to_index = {}
    set_of_L = set([L for mul, L in Rs])
    start = 0
    for L in set_of_L:
        L_to_index[L] = [start, start + 2 * L + 1]
        start += 2 * L + 1

    r = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j].reshape(3, -1)
    r = r.transpose(1, 0)
    r *= box_length / 2.
    r = torch.from_numpy(r)
    r_center = r.clone()

    if center is not None:
        r_center = r.clone()
        r_center[:,0] -= center[0]
        r_center[:,1] -= center[1]
        r_center[:,2] -= center[2]

    #Ys = self.sh(set_of_L, r)
    Ys = sh(set_of_L, r_center)
    #R = self.radial(r.norm(2, -1)).detach()  # [r_values, n_filters]
    R = radial(r_center.norm(2, -1)).detach()  # [r_values, n_filters]
    assert R.shape[-1] == mul_dim(Rs)

    R_helper = torch.zeros(R.shape[-1], dim(Rs))
    mul_start = 0
    y_start = 0
    Ys_indices = []
    for mul, L in Rs:
        Ys_indices += list(range(L_to_index[L][0], L_to_index[L][1])) * mul

    R_helper = map_mul_to_Rs(Rs)
    R_helper = R_helper.t()

    full_Ys = Ys[Ys_indices]  # [values, dim(Rs)]]  
    full_Ys = full_Ys.reshape(full_Ys.shape[0], -1)
    
    all_f = torch.einsum('xn,nd,dx->xd', R, R_helper, full_Ys)
    all_f = all_f.reshape(-1, all_f.shape[-1])
    return r, all_f
