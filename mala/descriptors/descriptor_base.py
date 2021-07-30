"""Base class for all descriptor calculators."""
import numpy as np
import ase

class DescriptorBase:
    """
    Base class for all descriptors available in MALA.

    Descriptors encode the atomic fingerprint of a DFT calculation.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.

    """

    def __init__(self, parameters):
        self.parameters = parameters.descriptors
        self.fingerprint_length = -1  # so iterations will fail
        self.dbg_grid_dimensions = parameters.debug.grid_dimensions

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert descriptors from a specified unit into the ones used in MALA.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.

        """
        raise Exception("No unit conversion method implemented for this"
                        " descriptor type.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert descriptors from MALA units into a specified unit.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.

        """
        raise Exception("No unit back conversion method implemented for "
                        "this descriptor type.")

    @staticmethod
    def enforce_pbc(atoms):
        """
        Explictly enforeces the PBC on an ASE atoms object.

        QE (and potentially other codes?) do that internally. Meaning that the
        raw positions of atoms (in Angstrom) can lie outside of the unit cell.
        When setting up the DFT calculation, these atoms get shifted into
        the unit cell. Since we directly use these raw positions for the
        descriptor calculation, we need to enforce that in the ASE atoms
        objects, the atoms are explicitly in the unit cell.

        Parameters
        ----------
        atoms : ase.atoms
            The ASE atoms object for which the PBC need to be enforced.

        Returns
        -------
        new_atoms : ase.atoms
            The ASE atoms object for which the PBC have been enforced.
        """
        new_atoms = atoms.copy()
        new_atoms.set_scaled_positions(new_atoms.get_scaled_positions())

        # This might be unecessary, but I think it is nice to have some sort of
        # metric here.
        rescaled_atoms = 0
        for i in range(0, len(atoms)):
            if False in (np.isclose(new_atoms[i].position,
                          atoms[i].position, atol=0.001)):
                rescaled_atoms += 1
        print("Descriptor calculation: had to enforce periodic boundary "
              "conditions on", rescaled_atoms, "atoms before calculation.")
        return new_atoms


