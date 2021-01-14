from .target_base import TargetBase
from .calculation_helpers import *
from scipy import integrate, interpolate
from scipy.optimize import toms748


class Density(TargetBase):
    """
    Electronic density.
    """
    def __init__(self, p):
        super(Density, self).__init__(p)
        # We operate on a per gridpoint basis. Per gridpoint, there is one value for the density (spin-unpolarized calculations).
        self.target_length = 1

    def get_number_of_electrons(self, density_data, grid_spacing_bohr=None, integration_method="summation"):
        """
        Calculates the number of electrons, from given density data.
        Input variables:
            - density_data: Electronic density on the given grid. Has to either be of the form
                    gridpoints
                            or
                    gridx x gridy x gridz.
            - integration method: Integration method used to integrate density on the grid.
            - grid_spacing_bohr: Grid spacing (in Bohr) used to construct this grid. As of now, only equidistant grids are supported.
        """

        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr

        # Check input data for correctness.
        data_shape = np.shape(density_data)
        if len(data_shape) != 3:
            if len(data_shape) != 1:
                raise Exception("Unknown Density shape, cannot calculate number of electrons.")
            elif integration_method != "summation":
                raise Exception("If using a 1D density array, you can only use summation as integration method.")


        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        if integration_method != "summation":
            number_of_electrons = density_data

            # X
            if data_shape[0] > 1:
                number_of_electrons = integrate_values_on_spacing(number_of_electrons, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                number_of_electrons = np.reshape(number_of_electrons, (data_shape[1], data_shape[2]))
                number_of_electrons *= grid_spacing_bohr

            # Y
            if data_shape[1] > 1:
                number_of_electrons = integrate_values_on_spacing(number_of_electrons, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                number_of_electrons = np.reshape(number_of_electrons, (data_shape[2]))
                number_of_electrons *= grid_spacing_bohr

            # Z
            if data_shape[2] > 1:
                number_of_electrons = integrate_values_on_spacing(number_of_electrons, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                number_of_electrons *= grid_spacing_bohr
        else:
            if len(data_shape) == 3:
                number_of_electrons = np.sum(density_data, axis=(0, 1, 2)) * (grid_spacing_bohr ** 3)
            if len(data_shape) == 1:
                number_of_electrons = np.sum(density_data, axis=0) * (grid_spacing_bohr ** 3)

        return number_of_electrons


    @classmethod
    def from_ldos(cls, ldos_object):
        """
        Create a Density calculator from an LDOS object.
        """
        return_density_object = Density(ldos_object.parameters)
        return_density_object.fermi_energy_eV = ldos_object.fermi_energy_eV
        return_density_object.temperature_K = ldos_object.temperature_K
        return_density_object.grid_spacing_Bohr = ldos_object.grid_spacing_Bohr
        return_density_object.number_of_electrons = ldos_object.number_of_electrons
        return_density_object.band_energy_dft_calculation = ldos_object.band_energy_dft_calculation
        return return_density_object
