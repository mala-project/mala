"""DOS calculation class."""
import os

from mala.targets.target import Target
from mala.targets.calculation_helpers import *
from scipy import integrate, interpolate
from scipy.optimize import toms748
from ase.units import Rydberg
import ase.io
from mala.common.parameters import printout


class DOS(Target):
    """Postprocessing / parsing functions for the density of states (DOS).

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    def __init__(self, params):
        super(DOS, self).__init__(params)
        self.target_length = self.parameters.ldos_gridsize

    def get_feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        return self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert the units of an array into the MALA units.

        MALA units for the DOS means 1/eV.

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
        if in_units == "1/eV":
            return array
        elif in_units == "1/Ry":
            return array * (1/Rydberg)
        else:
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

        MALA units for the DOS means 1/eV.

        Parameters
        ----------
        array : numpy.array
            Data in 1/eV.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            raise Exception("Unsupported unit for LDOS.")

    def read_from_qe_dos_txt(self, file_name, directory):
        """
        Read the DOS from a Quantum Espresso generated file.

        These files do not have a specified file ending, so I will call them
        qe.dos.txt here. QE saves the DOS in 1/eV.

        Parameters
        ----------
        file_name : string
            Name of the file containing the DOS.

        directory : string
            Directory containing the file file_name.

        Returns
        -------
        dos_data:
            DOS data in 1/eV.
        """
        # Create the desired/specified energy grid. We will use this to
        # check whether we have a correct file.

        energy_grid = self.get_energy_grid()
        return_dos_values = []

        # Open the file, then iterate through its contents.
        with open(os.path.join(directory, file_name), 'r') as infile:
            lines = infile.readlines()
            i = 0

            for dos_line in lines:
                # The first column contains the energy value.
                if "#" not in dos_line and i < self.parameters.ldos_gridsize:
                    e_val = float(dos_line.split()[0])
                    dosval = float(dos_line.split()[1])
                    if np.abs(e_val-energy_grid[i]) < self.parameters.\
                            ldos_gridspacing_ev*0.98:
                        return_dos_values.append(dosval)
                        i += 1

        return np.array(return_dos_values)

    def read_from_qe_out(self, path_to_file=None, smearing_factor=2):
        r"""
        Calculate the DOS from a Quantum Espresso DFT output file.

        The DOS will be read calculated via the eigenvalues and the equation

        .. math:: D(E) = \sum_i \sum_k w_k \delta(\epsilon-\epsilon_{ik})

        Parameters
        ----------
        path_to_file : string
            Path to the QE out file. If None, the QE output that was loaded
            via read_additional_calculation_data will be used.

        smearing_factor : int
            Smearing factor relative to the energy grid spacing. Default is 2.

        Returns
        -------
        dos_data:
            DOS data in 1/eV.
        """
        # dos_per_band = delta_f(e_grid,dft.eigs)
        if path_to_file is None:
            atoms_object = self.atoms
        else:
            atoms_object = ase.io.read(path_to_file, format="espresso-out")
        kweights = atoms_object.get_calculator().get_k_point_weights()
        if kweights is None:
            raise Exception("QE output file does not contain band information."
                            "Rerun calculation with verbosity set to 'high'.")

        # Get the gaussians for all energy values and calculate the DOS per
        # band.
        dos_per_band = gaussians(self.get_energy_grid(),
                                 atoms_object.get_calculator().
                                 band_structure().energies[0, :, :],
                                 smearing_factor*self.parameters.
                                 ldos_gridspacing_ev)
        dos_per_band = kweights[:, np.newaxis, np.newaxis]*dos_per_band

        # QE gives the band energies in eV, so no conversion necessary here.
        dos_data = np.sum(dos_per_band, axis=(0, 1))
        return dos_data

    def get_energy_grid(self):
        """
        Get energy grid.

        Returns
        -------
        e_grid : numpy.array
            Energy grid on which the DOS is defined.
        """
        emin = self.parameters.ldos_gridoffset_ev

        emax = self.parameters.ldos_gridoffset_ev + \
            self.parameters.ldos_gridsize * \
            self.parameters.ldos_gridspacing_ev
        grid_size = self.parameters.ldos_gridsize
        linspace_array = (np.linspace(emin, emax, grid_size, endpoint=False))
        return linspace_array

    def get_band_energy(self, dos_data, fermi_energy_eV=None,
                        temperature_K=None, integration_method="analytical"):
        """
        Calculate the band energy from given DOS data.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        energy_grid = self.get_energy_grid()
        return self.__band_energy_from_dos(dos_data, energy_grid,
                                           fermi_energy_eV, temperature_K,
                                           integration_method)

    def get_number_of_electrons(self, dos_data, fermi_energy_eV=None,
                                temperature_K=None,
                                integration_method="analytical"):
        """
        Calculate the number of electrons from given DOS data.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        energy_grid = self.get_energy_grid()
        return self.__number_of_electrons_from_dos(dos_data, energy_grid,
                                                   fermi_energy_eV,
                                                   temperature_K,
                                                   integration_method)

    def get_entropy_contribution(self, dos_data, fermi_energy_eV=None,
                                 temperature_K=None,
                                 integration_method="analytical"):
        """
        Calculate the entropy contribution to the total energy.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        entropy_contribution : float
            S/beta in eV.
        """
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        energy_grid = self.get_energy_grid()
        return self.\
            __entropy_contribution_from_dos(dos_data, energy_grid,
                                            fermi_energy_eV, temperature_K,
                                            integration_method)

    def get_self_consistent_fermi_energy_ev(self, dos_data,
                                            temperature_K=None,
                                            integration_method="analytical"):
        r"""
        Calculate the self-consistent Fermi energy.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        fermi_energy_self_consistent : float
            :math:`\epsilon_F` in eV.
        """
        # Parse the parameters.
        if temperature_K is None:
            temperature_K = self.temperature_K
        energy_grid = self.get_energy_grid()
        fermi_energy_sc = toms748(lambda fermi_sc:
                                  (self.
                                   __number_of_electrons_from_dos
                                   (dos_data, energy_grid,
                                    fermi_sc, temperature_K,
                                    integration_method)
                                   - self.number_of_electrons),
                                  a=energy_grid[0],
                                  b=energy_grid[-1])
        return fermi_energy_sc

    def get_density_of_states(self, dos_data):
        """Get the density of states."""
        return dos_data

    @classmethod
    def from_ldos(cls, ldos_object):
        """
        Create a DOS object from an LDOS object.

        Parameters
        ----------
        ldos_object : mala.targets.ldos.LDOS
            LDOS object used as input.

        Returns
        -------
        dos_object : DOS
            DOS object created from LDOS object.


        """
        return_dos_object = DOS(ldos_object.parameters)
        return_dos_object.fermi_energy_eV = ldos_object.fermi_energy_eV
        return_dos_object.temperature_K = ldos_object.temperature_K
        return_dos_object.voxel_Bohr = ldos_object.voxel_Bohr
        return_dos_object.number_of_electrons = ldos_object.number_of_electrons
        return_dos_object.band_energy_dft_calculation = \
            ldos_object.band_energy_dft_calculation
        return_dos_object.atoms = ldos_object.atoms
        return_dos_object.qe_input_data = ldos_object.qe_input_data
        return_dos_object.qe_pseudopotentials = ldos_object.qe_pseudopotentials
        return_dos_object.total_energy_dft_calculation = \
            ldos_object.total_energy_dft_calculation
        return_dos_object.kpoints = ldos_object.kpoints
        return_dos_object.number_of_electrons_from_eigenvals = \
            ldos_object.number_of_electrons_from_eigenvals

        return return_dos_object

    @staticmethod
    def __number_of_electrons_from_dos(dos_data, energy_grid, fermi_energy_eV,
                                       temperature_K, integration_method):
        """Calculate the number of electrons from DOS data."""
        # Calculate the energy levels and the Fermi function.

        fermi_vals = fermi_function(energy_grid, fermi_energy_eV,
                                    temperature_K)
        # Calculate the number of electrons.
        if integration_method == "trapz":
            number_of_electrons = integrate.trapz(dos_data * fermi_vals,
                                                  energy_grid, axis=-1)
        elif integration_method == "simps":
            number_of_electrons = integrate.simps(dos_data * fermi_vals,
                                                  energy_grid, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            number_of_electrons, abserr = integrate.quad(
                lambda e: dos_pointer(e) * fermi_function(e, fermi_energy_eV,
                                                          temperature_K),
                energy_grid[0], energy_grid[-1], limit=500,
                points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1",
                                                         fermi_energy_eV,
                                                         energy_grid,
                                                         temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return number_of_electrons

    @staticmethod
    def __band_energy_from_dos(dos_data, energy_grid,
                               fermi_energy_eV,
                               temperature_K, integration_method):
        """Calculate the band energy from DOS data."""
        # Calculate the energy levels and the Fermi function.
        fermi_vals = fermi_function(energy_grid, fermi_energy_eV,
                                    temperature_K)

        # Calculate the band energy.
        if integration_method == "trapz":
            band_energy = integrate.trapz(dos_data * (energy_grid *
                                                      fermi_vals),
                                          energy_grid, axis=-1)
        elif integration_method == "simps":
            band_energy = integrate.simps(dos_data * (energy_grid *
                                                      fermi_vals),
                                          energy_grid, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            band_energy, abserr = integrate.quad(
                lambda e: dos_pointer(e) * e * fermi_function(e,
                                                              fermi_energy_eV,
                                                              temperature_K),
                energy_grid[0], energy_grid[-1], limit=500,
                points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1",
                                                         fermi_energy_eV,
                                                         energy_grid,
                                                         temperature_K)
            band_energy_minus_uN = analytical_integration(dos_data, "F1", "F2",
                                                          fermi_energy_eV,
                                                          energy_grid,
                                                          temperature_K)
            band_energy = band_energy_minus_uN+fermi_energy_eV *\
                number_of_electrons
        else:
            raise Exception("Unknown integration method.")

        return band_energy

    @staticmethod
    def __entropy_contribution_from_dos(dos_data, energy_grid, fermi_energy_eV,
                                        temperature_K, integration_method):
        r"""
        Calculate the entropy contribution to the total energy from DOS data.

        More specifically, this gives -\beta^-1*S_S
        """
        # Calculate the entropy contribution to the energy.
        if integration_method == "trapz":
            multiplicator = entropy_multiplicator(energy_grid,
                                                  fermi_energy_eV,
                                                  temperature_K)
            entropy_contribution = integrate.trapz(dos_data * multiplicator,
                                                   energy_grid, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "simps":
            multiplicator = entropy_multiplicator(energy_grid, fermi_energy_eV,
                                                  temperature_K)
            entropy_contribution = integrate.simps(dos_data * multiplicator,
                                                   energy_grid, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            entropy_contribution, abserr = integrate.quad(
                lambda e: dos_pointer(e) *
                entropy_multiplicator(e, fermi_energy_eV, temperature_K),
                energy_grid[0], energy_grid[-1], limit=500,
                points=fermi_energy_eV)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "analytical":
            entropy_contribution = analytical_integration(dos_data, "S0", "S1",
                                                          fermi_energy_eV,
                                                          energy_grid,
                                                          temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return entropy_contribution
