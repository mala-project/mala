"""Electronic density calculation class."""
from ase.units import Rydberg, Bohr

from .target import Target


class AtomicForce(Target):
    """Postprocessing / parsing functions for atomic forces.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    def __init__(self, params):
        """
        Create a Density object.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this TargetBase object.

        """
        super(AtomicForce, self).__init__(params)

    def get_feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        return 3

    @staticmethod
    def convert_units(array, in_units="eV/Ang"):
        """
        Convert the units of an array into the MALA units.

        MALA units for the LDOS means 1/eV.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array. Currently supported are:

                 - 1/eV (no conversion, MALA unit)
                 - 1/Ry

        Returns
        -------
        converted_array : numpy.array
            Data in 1/eV.
        """
        if in_units == "eV/Ang":
            return array
        elif in_units == "Ry/Bohr":
            return array * (Rydberg/Bohr)
        else:
            raise Exception("Unsupported unit for atomic forces.")
