import os

import mala
from ase.io import read
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from mala.datahandling.data_repo import data_repo_path
atoms_path = os.path.join(os.path.join(data_repo_path, "Al36"),
                          "Al.pw.scf.out")
"""
ex16_observables.py : Shows which observables MALA can calculate upon user
request. Not specifically shown are number of electrons, total energy and
band energy, as these have been showcased in the other examples.
"""


def plot_radial_distribution_function(show_plot=True):
    atoms = read(atoms_path, format="espresso-out")

    # Any target can be used.
    # Note: The function implemented here differs from the ASE implementation.
    # It takes periodic boundary condition into account.
    rdf, radii = mala.LDOS.radial_distribution_function_from_atoms(atoms, number_of_bins=100)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(radii, rdf)
    if show_plot:
        plt.show()


def plot_three_particle_correlation_function(show_plot=True):
    atoms = read(atoms_path, format="espresso-out")

    # Any target can be used.
    nbins = 20
    tpcf = mala.LDOS.three_particle_correlation_function_from_atoms(atoms, nbins, rMax=5.0)

    # For the plotting of the contour plots, we define a custom color map.
    # plot contour plots.
    cmap = pl.cm.get_cmap("viridis")
    viridis_transparent = cmap(np.arange(cmap.N))
    viridis_transparent[:,-1] = np.linspace(0.2, 1.0, cmap.N)
    viridis_transparent = ListedColormap(viridis_transparent)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(tpcf[1][0, :, 0, 0], tpcf[1][1, 0, :, 0])

    # Not all combinations of radii # of the TPCF are enough for a contourplot.
    minval = 8
    for i in range(minval, nbins):
        ax.contourf(X, Y, tpcf[0][:, :, i], offset=tpcf[1][2, 0, 0, i],
                    cmap=viridis_transparent)
    ax.set_zlim3d(tpcf[1][2, 0, 0, minval]-0.01,
                  tpcf[1][2, 0, 0, nbins-1]+0.01)
    if show_plot:
        plt.show()


def plot_static_structure_factor(show_plot=True):
    atoms = read(atoms_path, format="espresso-out")

    # Any target can be used.
    # Note: The function implemented here differs from the ASE implementation.
    # It takes periodic boundary condition into account.
    static_structure, kpoints = mala.LDOS.static_structure_factor_from_atoms(atoms, 100, 12)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(kpoints, static_structure)
    if show_plot:
        plt.show()


# Set show_plot to True for those quantities you would like to see.
plot_radial_distribution_function(show_plot=False)
plot_three_particle_correlation_function(show_plot=False)
plot_static_structure_factor(show_plot=False)
