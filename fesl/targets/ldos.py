from .cube_parser import read_cube
from .target_base import TargetBase
from .calculation_helpers import *
from .dos import DOS
import numpy as np
import math
from ase.units import Rydberg


class LDOS(TargetBase):
    """Local density of states.
    Evaluation follow the outline and code provided in/by [1].
    """
    def __init__(self, p):
        super(LDOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Converts the units of an LDOS array. Can be used e.g. for post processing purposes.
        This function always returns the LDOS in 1/eV.
        """
        if in_units == "1/eV":
            return array
        elif in_units == "1/Ry":
            return array / Rydberg
        else:
            print(in_units)
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Converts the units of an LDOS array back from internal use to outside use.
        Can be used e.g. for post processing purposes.
        This function always accepts the LDOS in 1/eV and outputs them in the desired unit.
        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            print(out_units)
            raise Exception("Unsupported unit for LDOS.")


    def read_from_cube(self, file_name_scheme, directory):
        """Reads the LDOS data from multiple cube files located in the snapshot directory."""

        # First determine how many digits the last file in the list of LDOS.cube files
        # will have.
        # QuantumEspresso saves the pp.x cube files like this:
        # tmp.pp001ELEMENT_ldos.cube
        # tmp.pp002ELEMENT_ldos.cube
        # tmp.pp003ELEMENT_ldos.cube
        # ...
        # tmp.pp100ELEMENT_ldos.cube

        digits = int(math.log10(self.parameters.ldos_gridsize))+1

        # Iterate over the amount of specified LDOS input files.
        # QE is a Fortran code, so everything is 1 based.
        for i in range(1, self.parameters.ldos_gridsize+1):
            tmp_file_name = file_name_scheme
            tmp_file_name = tmp_file_name.replace("*", str(i).zfill(digits))

            # Open the cube file
            data, meta = read_cube(directory+tmp_file_name)
            print(data)
            quit()

        # print(dir+file_name_scheme)
        # ldosfilelist = glob.glob(dir+file_name_scheme)
        # print(ldosfilelist)

    def calculate_energy(self, ldos_data):
        """Calculates the total energy from given ldos_data. The ldos_data can either
        have the form

        gridpoints x energygrid

        or

        gridx x gridy x gridz x energygrid.
        The total energy is calculated as:

        E_tot[D](R) = E_b[D] - S_s[D]/beta - U[D] + E_xc[D]-V_xc[D]+V^ii(R)
        """

    def get_band_energy(self, ldos_data, fermi_energy_eV=None, temperature_K=None, grid_spacing_bohr=None, integration_method="simps"):
        """
        Calculates the band energy, from given LDOS data.
        Input variables:
            - ldos_data - This method can only be called if the LDOS data is presented in the form:
                    gridx x gridy x gridz x energygrid.
            - integration_method: Integration method to be used, can either be "trapz" for trapezoid or "simps" for Simpson method.
        """

        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr

        # The band energy is calculated using the DOS.
        dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr, integration_method=integration_method)

        # We further process the DOS by using a static function of the DOS class.
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        nr_energy_levels = self.parameters.ldos_gridsize
        band_energy = DOS.band_energy_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV, temperature_K,
                                               integration_method=integration_method)

        return band_energy

    def get_number_of_electrons(self, ldos_data, grid_spacing_bohr=None, fermi_energy_eV=None, temperature_K=None, integration_method="simps"):
        """
        Calculates the number of electrons, from given DOS data.
        Input variables:
            - ldos_data - This method can only be called if the LDOS data is presented in the form:
                    gridx x gridy x gridz x energygrid.
        """
        # The number of electrons is calculated using the DOS.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr


        dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr, integration_method=integration_method)

        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        nr_energy_levels = self.parameters.ldos_gridsize
        return DOS.number_of_electrons_from_dos(dos_data, emin, emax, nr_energy_levels, fermi_energy_eV,
                             temperature_K, integration_method)


    def get_density(self, ldos_data, fermi_energy_ev=None, temperature_K=None, conserve_dimensions=False, integration_method="simps"):
        """
        Calculates the electronic density, from given LDOS data.
        Input variables:
            - ldos_data - can either have the form
                    gridpoints x energygrid
                            or
                    gridx x gridy x gridz x energygrid.
            - conserve_dimensions: If true, gridx x gridy x gridz x energygrid data will be transformed into data
                of dimensions gridx x gridy x gridz. If false, no such transformation will be done, and the result will be in
                the shape of gridpoints.
            - integration_method: Integration method to be used, can either be "trapz" for trapezoid or "simps" for Simpson method.
        """
        if fermi_energy_ev is None:
            fermi_energy_ev = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        ldos_data_shape = np.shape(ldos_data)
        if len(ldos_data_shape) == 2:
            # We have the LDOS as gridpoints x energygrid, so no further operation is necessary.
            ldos_data_used = ldos_data
            pass
        elif len(ldos_data_shape) == 4:
            # We have the LDOS as gridx x gridy x gridz x energygrid, so some reshaping needs to be done.
            ldos_data_used = ldos_data.reshape([ldos_data_shape[0]*ldos_data_shape[1]*ldos_data_shape[2], ldos_data_shape[3]])
            # We now have the LDOS as gridpoints x energygrid.

        else:
            raise Exception("Invalid LDOS array shape.")

        # Parse the information about the energy grid.
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        nr_elvls = self.parameters.ldos_gridsize

        # Build the energy grid and calculate the fermi function.
        energy_values = np.linspace(emin, emax, nr_elvls)
        fermi_values = fermi_function(energy_values, fermi_energy_ev, temperature_K, energy_units="eV")

        density_values = integrate_values_on_grid(ldos_data_used * (energy_values * fermi_values), energy_values, axis=-1,
                                                  method=integration_method)

        if len(ldos_data_shape) == 4 and conserve_dimensions is True:
            ldos_data_shape = list(ldos_data_shape)
            ldos_data_shape[-1] = 1
            density_values = density_values.reshape(ldos_data_shape)

        return density_values

    def get_density_of_states(self, ldos_data, grid_spacing_bohr=None, integration_method="simps"):
        """Calculates the density of states, from given LDOS data.
        Input variables:
            - ldos_data - This method can only be called if the LDOS data is presented in the form:
                    gridx x gridy x gridz x energygrid
            - integration_method: Integration method to be used:
                    - can either be "trapz" for trapezoid or "simps" for Simpson method or "summation" for a
                    simple "integration by summation".
        """

        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr

        ldos_data_shape = np.shape(ldos_data)
        if len(ldos_data_shape) != 4:
            raise Exception("Invalid LDOS array shape (correct: x,y,z,energy)")

        # We have the LDOS as gridx x gridy x gridz x energygrid, no further operation is necessary.
        dos_values = ldos_data#.copy()

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        if integration_method != "summation":
            # X
            if ldos_data_shape[0] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0, method=integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[1], ldos_data_shape[2], ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr

            # Y
            if ldos_data_shape[1] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0, method=integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[2], ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr

            # Z
            if ldos_data_shape[2] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0, method=integration_method)
            else:
                dos_values = np.reshape(dos_values, ldos_data_shape[3])
                dos_values *= grid_spacing_bohr
        else:
            dos_values = np.sum(ldos_data, axis=(0, 1, 2)) * (grid_spacing_bohr**3)

        return dos_values

