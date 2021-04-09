# +
import math
import numpy as np
import torch

from e3nn.util.cache_file import cached_dirpklgz
import e3nn.o3
import e3nn.util.plot as plot
#from se3cnn.util.cache_file import cached_dirpklgz
#import se3cnn.SO3
#import se3cnn.util.plot as plot
# -

__author__  = "Tess E. Smidt"

@cached_dirpklgz("cache/euler_grids")
def euler_angles_on_grid(n):
    alpha = torch.linspace(0, 2 * math.pi, 2 * n)
    beta = torch.linspace(0, math.pi, n)
    gamma = torch.linspace(0, 2 * math.pi, 2 * n)
    alpha, beta, gamma = torch.meshgrid(alpha, beta, gamma)
    return alpha, beta, gamma


def spherical_surface(n):
    alpha = torch.linspace(0, 2 * math.pi, 2 * n)
    beta = torch.linspace(0, math.pi, 2 * n)
    beta, alpha = torch.meshgrid(beta, alpha)
    x, y, z = e3nn.SO3.angles_to_xyz(alpha, beta)
    return x, y, z, alpha, beta


@cached_dirpklgz("cache/sh_grids")
def spherical_harmonics_on_grid(L, n):
    x, y, z, alpha, beta = spherical_surface(n)
    print(x.shape, alpha.shape)
    #return x, y, z, se3cnn.SO3.spherical_harmonics(L, alpha, beta)
    return x, y, z, e3nn.o3.spherical_harmonics(L, alpha, beta)
    


@cached_dirpklgz("cache/wigner_D_grids")
def wigner_D_on_grid(L, n):
    alpha, beta, gamma = euler_angles_on_grid(n)
    shape = alpha.shape
    abc = torch.stack([alpha.flatten(), beta.flatten(), gamma.flatten()],
                      dim=-1)
    wig_Ds = []
    for a, b, c in abc:
        #wig_Ds.append(se3cnn.SO3.irr_repr(L, a, b, c))
        wig_Ds.append(e3nn.o3.irr_repr(L, a, b, c))
    wig_D_shape = wig_Ds[0].shape
    return torch.stack(wig_Ds).reshape(shape + wig_D_shape)



