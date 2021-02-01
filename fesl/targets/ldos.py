from .cube_parser import read_cube
from .target_base import TargetBase
from .calculation_helpers import *
from .dos import DOS
from .density import Density
import numpy as np
import math
from ase.units import Rydberg


class LDOS(TargetBase):
    """Local density of states.
    Evaluation follow the outline and code provided in/by [1].
    """

    def __init__(self, p):
        """Constructor.

        Returns a LDOS object.

        Parameters
        ----------
        p : fesl.common.parameters.Parameters
            A Parameters object.

        Returns
        -------
        fesl.targets.ldos.LDOS
            A LDOS object.
        """
        super(LDOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """Converts units of an array to FESL units.

        Converts the units of a given array to FESL units, i.e. 1/eV.

        Parameters
        ----------
        array : numpy array or list
            The array the unit conversion is performed upon.
        in_units : str
            The units of the original array. Default is "1/eV". Supported units:
                "1/eV" (no unit conversion is done, as this is the FESL default unit as well.)
                "1/Ry"

        Returns
        -------
        numpy array or list
            Array in the units 1/eV.

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
        """Converts units of array from FESL units to desired units.

        Converts the units of a given array from FESL units (i.e. 1/eV) to desired units, e.g. 1/Ry

        Parameters
        ----------
        array : numpy array or list
            The array the unit conversion is performed upon.
        out_units : str
            The units one wants to the array to convert to. Supported units:
                "1/eV" (no unit conversion is done, as this is the FESL default unit as well.)
                "1/Ry"

        Returns
        -------
        numpy array or list
            Array in the units specified by out_unit.

        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            print(out_units)
            raise Exception("Unsupported unit for LDOS.")

    def read_from_cube(self, file_name_scheme, directory, ldos_units="1/eV"):
        """Reads the LDOS data from multiple cube files located in the snapshot directory."""

        # First determine how many digits the last file in the list of LDOS.cube files
        # will have.
        # QuantumEspresso saves the pp.x cube files like this:
        # tmp.pp001ELEMENT_ldos.cube
        # tmp.pp002ELEMENT_ldos.cube
        # tmp.pp003ELEMENT_ldos.cube
        # ...
        # tmp.pp100ELEMENT_ldos.cube

        # Find out the number of digits that are needed to encode this grid (by QE).
        digits = int(math.log10(self.parameters.ldos_gridsize)) + 1


        # Iterate over the amount of specified LDOS input files.
        # QE is a Fortran code, so everything is 1 based.
        print("Reading "+str(self.parameters.ldos_gridsize)+" LDOS files from"+directory+".")
        for i in range(1, self.parameters.ldos_gridsize + 1):
            tmp_file_name = file_name_scheme
            tmp_file_name = tmp_file_name.replace("*", str(i).zfill(digits))

            # Open the cube file#
            data, meta = read_cube(directory + tmp_file_name)

            # Once we have read the first cube file, we know the dimensions of the LDOS and can prepare the array
            # in which we want to store the LDOS.
            if i == 1:
                data_shape = np.shape(data)
                ldos_data = np.zeros((data_shape[0], data_shape[1], data_shape[2], self.parameters.ldos_gridsize), dtype=np.float)

            # Convert and then append the LDOS data.
            data = self.convert_units(data, in_units=ldos_units)
            ldos_data[:, :, :, i-1] = data[:, :, :]
        return ldos_data


    def get_energy_grid(self):
        return np.arange(self.parameters.ldos_gridoffset_ev, self.parameters.ldos_gridoffset_ev+self.parameters.ldos_gridsize*self.parameters.ldos_gridspacing_ev,self.parameters.ldos_gridspacing_ev)

    def get_total_energy(self, ldos_data=None, dos_data=None, density_data=None, fermi_energy_eV=None, temperature_K=None,
                        grid_spacing_bohr=None, grid_integration_method="summation",
                        energy_integration_method="analytical", shift_energy_grid=True, atoms_Angstrom=None,
                         qe_input_data=None, qe_pseudopotentials=None):
        """
        Calculates the total energy either from given LDOS data or given DOS and density data.
        """

        # Get relevant values from DFT calculation, if not otherwise specified.
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        # Check the input.
        if ldos_data is None:
            if dos_data is None or density_data is None:
                raise Exception("No input data provided to caculate total energy. Provide EITHER LDOS OR DOS and density.")

        # Calculate DOS data if need be.
        if dos_data is None:
            dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr=grid_spacing_bohr,
                                                  integration_method=grid_integration_method)

        # Calculate density data if need be.
        if density_data is None:
            density_data = self.get_density(ldos_data, fermi_energy_ev=fermi_energy_eV, temperature_K=temperature_K,
                                             integration_method=energy_integration_method, shift_energy_grid=shift_energy_grid)

        # Now we can create calculation objects to get the necessary quantities.
        dos_calculator = DOS.from_ldos(self)
        density_calculator = Density.from_ldos(self)

        # With these calculator objects we can calculate all the necessary quantities to construct the total energy.
        # (According to Eq. 9 in [1])
        # Band energy (kinetic energy)
        e_band = dos_calculator.get_band_energy(dos_data, fermi_energy_eV=fermi_energy_eV, temperature_K=temperature_K,
                                                integration_method=energy_integration_method,
                                                shift_energy_grid=shift_energy_grid)

        # Smearing / Entropy contribution
        e_entropy_contribution = dos_calculator.get_entropy_contribution(dos_data,fermi_energy_eV=fermi_energy_eV,
                                                                         temperature_K=temperature_K,
                                                                         integration_method=energy_integration_method,
                                                                         shift_energy_grid=shift_energy_grid)

        # Density based energy contributions (via QE)
        e_rho_times_v_hxc, e_hartree,  e_xc, e_ewald \
            = density_calculator.\
            get_energy_contributions(density_data, qe_input_data=qe_input_data, atoms_Angstrom=atoms_Angstrom,
                                     qe_pseudopotentials=qe_pseudopotentials)
        e_total = e_band + e_rho_times_v_hxc + e_hartree + e_xc + e_ewald + e_entropy_contribution

        return e_total




    def get_band_energy(self, ldos_data, fermi_energy_eV=None, temperature_K=None, grid_spacing_bohr=None,
                        grid_integration_method="summation", energy_integration_method="analytical",
                        shift_energy_grid=True):
        """
        Calculates the band energy, from given LDOS data.
        Input variables:
            - ldos_data - This method can only be called if the LDOS data is presented in the form:
                    gridx x gridy x gridz x energygrid.
            - integration_method: Integration method to be used, can either be "trapz" for trapezoid or "simps" for Simpson method.
        """

        # The band energy is calculated using the DOS.
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr
        dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr, integration_method=grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate the band energy..
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.get_band_energy(dos_data, fermi_energy_eV=fermi_energy_eV,
                                              temperature_K=temperature_K,
                                              integration_method=energy_integration_method,
                                              shift_energy_grid=shift_energy_grid)

    def get_number_of_electrons(self, ldos_data, grid_spacing_bohr=None, fermi_energy_eV=None, temperature_K=None,
                                grid_integration_method="summation", energy_integration_method="analytical",
                                shift_energy_grid=True):
        """
        Calculates the number of electrons, from given DOS data.
        Input variables:
            - ldos_data - This method can only be called if the LDOS data is presented in the form:
                    gridx x gridy x gridz x energygrid.
        """

        # The number of electrons is calculated using the DOS.
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr
        dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr, integration_method=grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate the number of electrons.
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.get_number_of_electrons(dos_data, fermi_energy_eV=fermi_energy_eV,
                                                      temperature_K=temperature_K,
                                                      integration_method=energy_integration_method,
                                                      shift_energy_grid=shift_energy_grid)

    def get_self_consistent_fermi_energy_ev(self, ldos_data, grid_spacing_bohr=None, fermi_energy_eV=None, temperature_K=None,
                                grid_integration_method="summation", energy_integration_method="analytical",
                                shift_energy_grid=True):
        """
        Calculates the "self-consistent Fermi energy", i.e. the fermi energy for which integrating over the DOS
        gives the EXACT number of atoms.
        """

        # The Fermi energy is calculated using the DOS.
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr
        dos_data = self.get_density_of_states(ldos_data, grid_spacing_bohr, integration_method=grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate the number of electrons.
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.get_self_consistent_fermi_energy_ev(dos_data,
                                                      temperature_K=temperature_K,
                                                      integration_method=energy_integration_method,
                                                      shift_energy_grid=shift_energy_grid)


    def get_density(self, ldos_data, fermi_energy_ev=None, temperature_K=None, conserve_dimensions=False,
                    integration_method="analytical", shift_energy_grid=True):
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
            ldos_data_used = ldos_data.reshape(
                [ldos_data_shape[0] * ldos_data_shape[1] * ldos_data_shape[2], ldos_data_shape[3]])
            # We now have the LDOS as gridpoints x energygrid.

        else:
            raise Exception("Invalid LDOS array shape.")

        # Parse the information about the energy grid.
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize * self.parameters.ldos_gridspacing_ev + self.parameters.ldos_gridoffset_ev
        energy_grid_spacing = self.parameters.ldos_gridspacing_ev

        # Build the energy grid and calculate the fermi function.
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_values = fermi_function(energy_vals, fermi_energy_ev, temperature_K, energy_units="eV")

        # Calculate the number of electrons.
        if integration_method == "trapz":
            density_values = integrate.trapz(ldos_data_used * fermi_values, energy_vals, axis=-1)
        elif integration_method == "simps":
            density_values = integrate.simps(ldos_data_used * fermi_values, energy_vals, axis=-1)
        elif integration_method == "analytical":
            density_values = analytical_integration(ldos_data_used, "F0", "F1", fermi_energy_ev, energy_vals,
                                                    temperature_K)
        else:
            raise Exception("Unknown integration method.")

        if len(ldos_data_shape) == 4 and conserve_dimensions is True:
            ldos_data_shape = list(ldos_data_shape)
            ldos_data_shape[-1] = 1
            density_values = density_values.reshape(ldos_data_shape)

        return density_values

    def get_density_of_states(self, ldos_data, grid_spacing_bohr=None, integration_method="summation"):
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
            if len(ldos_data_shape) != 2:
                raise Exception("Unknown LDOS shape, cannot calculate DOS.")
            elif integration_method != "summation":
                raise Exception("If using a 2D LDOS array, you can only use summation as integration method.")

        # We have the LDOS as gridx x gridy x gridz x energygrid, no further operation is necessary.
        dos_values = ldos_data  # .copy()

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        if integration_method != "summation":
            # X
            if ldos_data_shape[0] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[1], ldos_data_shape[2], ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr

            # Y
            if ldos_data_shape[1] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[2], ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr

            # Z
            if ldos_data_shape[2] > 1:
                dos_values = integrate_values_on_spacing(dos_values, grid_spacing_bohr, axis=0,
                                                         method=integration_method)
            else:
                dos_values = np.reshape(dos_values, ldos_data_shape[3])
                dos_values *= grid_spacing_bohr
        else:
            if len(ldos_data_shape) == 4:
                dos_values = np.sum(ldos_data, axis=(0, 1, 2)) * (grid_spacing_bohr ** 3)
            if len(ldos_data_shape) == 2:
                dos_values = np.sum(ldos_data, axis=0) * (grid_spacing_bohr ** 3)

        return dos_values
