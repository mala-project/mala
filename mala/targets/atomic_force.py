"""Electronic density calculation class."""
from .target_base import TargetBase
from .calculation_helpers import *
import warnings
from scipy.constants import physical_constants
from scipy.constants import e
from scipy.constants import epsilon_0
from scipy.constants import pi
from scipy.constants import Rydberg
from scipy.constants import h
from scipy.constants import c
from mala.common.parameters import printout


class AtomicForce(TargetBase):
    def __init__(self, params):
        """
        Create a Density object.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this TargetBase object.

        """
        super(AtomicForce, self).__init__(params)
        # We operate on a per gridpoint basis. Per gridpoint,
        # there is one value for the density (spin-unpolarized calculations).
        self.target_length = 3

    @staticmethod
    def get_hellman_feynman_factor():
        """
        Calculate the prefactor for the Hellman-Feynman forces.

        Returns
        -------
        """

        prefactor_J_m = (e*e / (4 * pi * epsilon_0))
        Ry_in_Joule = Rydberg*h*c
        # This should be the correct formula but it doesn't work yet.
        prefactor_Ry_Bohr = prefactor_J_m*(physical_constants["Bohr radius"][0]/Ry_in_Joule)
        final_prefactor = prefactor_Ry_Bohr / (
                    physical_constants["Bohr radius"][0] *
                    physical_constants["Bohr radius"][0])
        return final_prefactor