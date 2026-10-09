"""Tests for the scalar return value of the electron-number target methods."""

import json

import numpy as np
from ase.cell import Cell

import mala
from mala.targets.density import Density


def _voxel():
    """A unit cell whose volume makes the summation result easy to read."""
    return Cell.fromcellpar([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])


class TestDensityNumberOfElectrons:
    """
    Density.get_number_of_electrons must return a scalar, not a 0-d array.

    A 0-d np.ndarray is unhashable, fails isinstance(x, float) and cannot be
    serialised with json, unlike the equivalent DOS/LDOS implementations and
    the *number_of_electrons* attributes on the target classes.
    """

    def test_get_number_of_electrons_is_scalar(self):
        density = Density(mala.Parameters())
        density_data = np.ones((2, 2, 2, 1))

        number_of_electrons = density.get_number_of_electrons(
            density_data=density_data,
            voxel=_voxel(),
            integration_method="summation",
        )

        assert isinstance(number_of_electrons, np.float64)
        assert not isinstance(number_of_electrons, np.ndarray)
        assert number_of_electrons == 8.0
        assert json.dumps(number_of_electrons) == "8.0"

    def test_get_number_of_electrons_is_scalar_for_a_2d_grid(self):
        density = Density(mala.Parameters())
        density_data = np.ones((4, 1))

        number_of_electrons = density.get_number_of_electrons(
            density_data=density_data,
            voxel=_voxel(),
            integration_method="summation",
        )

        assert isinstance(number_of_electrons, np.float64)
        assert not isinstance(number_of_electrons, np.ndarray)
        assert number_of_electrons == 4.0

    def test_number_of_electrons_property_is_scalar(self):
        density = Density(mala.Parameters())
        density.voxel = _voxel()
        density.density = np.ones((2, 2, 2, 1))

        number_of_electrons = density.number_of_electrons

        assert isinstance(number_of_electrons, np.float64)
        assert not isinstance(number_of_electrons, np.ndarray)
        assert number_of_electrons == 8.0
