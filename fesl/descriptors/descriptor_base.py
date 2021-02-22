class DescriptorBase:
    """
    Base class for all descriptors available in FESL.

    Descriptors encode the atomic fingerprint of a DFT calculation.
    """

    def __init__(self, parameters):
        """
        Create a DescriptorBase object.

        Parameters
        ----------
        parameters : fesl.common.parameters.Parameters
            Parameters object used to create this object.
        """
        self.parameters = parameters.descriptors
        self.fingerprint_length = -1  # so iterations will fail
        self.dbg_grid_dimensions = parameters.debug.grid_dimensions

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert descriptors from a specified unit into the ones used in FESL.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in FESL units.

        """
        raise Exception("No unit conversion method implemented for this"
                        " descriptor type.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert descriptors from FESL units into a specified unit.

        Parameters
        ----------
        array : numpy.array
            Data in FESL units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.

        """
        raise Exception("No unit back conversion method implemented for "
                        "this descriptor type.")
