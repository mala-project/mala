"""DOS calculation class."""

from functools import cached_property

import ase.io
from ase.units import Rydberg, J
import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import toms748

from mala.common.parameters import printout
from mala.common.parallelizer import get_rank, barrier, get_comm
from mala.targets.target import Target
from mala.targets.calculation_helpers import (
    fermi_function,
    gaussians,
    analytical_integration,
    get_beta,
    entropy_multiplicator,
   analytical_integration_weights,
)


class DOS(Target):
    """
    Postprocessing / parsing functions for the density of states (DOS).

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    ##############################
    # Constructors
    ##############################

    def __init__(self, params):
        super(DOS, self).__init__(params)
        self.density_of_states = None

    @classmethod
    def from_ldos_calculator(cls, ldos_object):
        """
        Create a DOS object from an LDOS object.

        If the LDOS object has data associated with it, this data will
        be copied.

        Parameters
        ----------
        ldos_object : mala.targets.ldos.LDOS
            LDOS object used as input.

        Returns
        -------
        dos_object : mala.targets.dos.DOS
            DOS object created from LDOS object.
        """
        return_dos_object = DOS(ldos_object.parameters)
        return_dos_object.fermi_energy_dft = ldos_object.fermi_energy_dft
        return_dos_object.temperature = ldos_object.temperature
        return_dos_object.voxel = ldos_object.voxel
        return_dos_object.number_of_electrons_exact = (
            ldos_object.number_of_electrons_exact
        )
        return_dos_object.band_energy_dft_calculation = (
            ldos_object.band_energy_dft_calculation
        )
        return_dos_object.atoms = ldos_object.atoms
        return_dos_object.qe_input_data = ldos_object.qe_input_data
        return_dos_object.qe_pseudopotentials = ldos_object.qe_pseudopotentials
        return_dos_object.total_energy_dft_calculation = (
            ldos_object.total_energy_dft_calculation
        )
        return_dos_object.kpoints = ldos_object.kpoints
        return_dos_object.number_of_electrons_from_eigenvals = (
            ldos_object.number_of_electrons_from_eigenvals
        )
        return_dos_object.local_grid = ldos_object.local_grid
        return_dos_object._parameters_full = ldos_object._parameters_full

        # If the source calculator has LDOS data, then this new object
        # can have DOS data.
        if ldos_object.local_density_of_states is not None:
            return_dos_object.density_of_states = ldos_object.density_of_states

        return return_dos_object

    @classmethod
    def from_numpy_file(cls, params, path, units="1/eV"):
        """
        Create an DOS calculator from a numpy array saved in a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        path : string
            Path to file that is being read.

        units : string
            Units the DOS is saved in.

        Returns
        -------
        dos_calculator : mala.targets.dos.DOS
            DOS calculator object.
        """
        return_dos = DOS(params)
        return_dos.read_from_numpy_file(path, units=units)
        return return_dos

    @classmethod
    def from_numpy_array(cls, params, array, units="1/eV"):
        """
        Create an DOS calculator from a numpy array in memory.

        By using this function rather then setting the local_density_of_states
        object directly, proper

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        array : numpy.ndarray
            Path to file that is being read.

        units : string
            Units the DOS is saved in.

        Returns
        -------
        dos_calculator : mala.targets.dos.DOS
            DOS calculator object.
        """
        return_dos = DOS(params)
        return_dos.read_from_array(array, units=units)
        return return_dos

    @classmethod
    def from_qe_dos_txt(cls, params, path):
        """
        Create a DOS calculator from a Quantum Espresso generated file.

        These files do not have a specified file ending, so I will call them
        qe.dos.txt here. QE saves the DOS in 1/eV.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        path : string
            Path of the file containing the DOS.

        Returns
        -------
        dos_calculator : mala.targets.dos.DOS
            DOS calculator object.
        """
        return_dos = DOS(params)
        return_dos.read_from_qe_dos_txt(path)
        return return_dos

    @classmethod
    def from_qe_out(cls, params, path):
        """
        Create a DOS calculator from a Quantum Espresso output file.

        This will only work if the QE calculation has been performed with
        very verbose output and contains the bands at all k-points.
        As much information will be read from the QE output files as possible.
        There is no need to call read_additional_calculation_data afterwards.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        path : string
            Path of the file containing the DOS.

        Returns
        -------
        dos_calculator : mala.targets.dos.DOS
            DOS calculator object.
        """
        return_dos = DOS(params)

        # In this case, we may just read the entire qe.out file.
        return_dos.read_additional_calculation_data(path, "espresso-out")

        # This method will use the ASE atoms object read above automatically.
        return_dos.read_from_qe_out()
        return return_dos

    ##############################
    # Properties
    ##############################

    @property
    def feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        if isinstance(self.parameters.ldos_gridsize, int):
            return self.parameters.ldos_gridsize
        elif isinstance(self.parameters.ldos_gridsize, list):
            # For splits, we sum up the individual grid sizes, BUT we
            # have to subtract one for each split, as the last energy
            # of each section gets discarded. So for three sections,
            # we have to subtract 2.
            return (
                np.sum(self.parameters.ldos_gridsize)
                - len(self.parameters.ldos_gridsize)
                + 1
            )

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "DOS"

    @property
    def si_unit_conversion(self):
        """
        Numeric value of the conversion from MALA (ASE) units to SI.

        Needed for OpenPMD interface.
        """
        return J

    @property
    def si_dimension(self):
        """Dictionary containing the SI unit dimensions in OpenPMD format."""
        import openpmd_api as io

        return {
            io.Unit_Dimension.M: -1,
            io.Unit_Dimension.L: -2,
            io.Unit_Dimension.T: 2,
        }

    @property
    def density_of_states(self):
        """Density of states as array."""
        return self._density_of_states

    @density_of_states.setter
    def density_of_states(self, new_dos):
        self._density_of_states = new_dos
        # Setting a new DOS means we have to uncache priorly cached
        # properties.
        self.uncache_properties()

    def get_target(self):
        """
        Get the target quantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.

        Returns
        -------
        density_of_states : numpy.ndarray
            Electronic density of states.

        """
        return self.density_of_states

    def invalidate_target(self):
        """
        Invalidates the saved target wuantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        self.density_of_states = None

    @cached_property
    def energy_grid(self):
        """Energy grid on which the DOS is expressed."""
        return self._get_energy_grid()

    @cached_property
    def band_energy(self):
        """Band energy of the system, calculated via cached DOS."""
        if self.density_of_states is not None:
            return self.get_band_energy()
        else:
            raise Exception(
                "No cached DOS available to calculate this property."
            )

    @cached_property
    def d_band_energy_d_dos(self):
        """Derivative of band energy, calculated via cached DOS."""
        if self.density_of_states is not None:
            return self.get_d_band_energy_d_dos()
        else:
            raise Exception("No cached DOS available to calculate this "
                            "property.")

    @cached_property
    def number_of_electrons(self):
        """
        Number of electrons in the system, calculated via cached DOS.

        Does not necessarily match up exactly with KS-DFT provided values,
        due to discretization errors.
        """
        if self.density_of_states is not None:
            return self.get_number_of_electrons()
        else:
            raise Exception(
                "No cached DOS available to calculate this property."
            )

    @cached_property
    def fermi_energy(self):
        """
        "Self-consistent" Fermi energy of the system.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated. Calculated via cached DOS.
        """
        if self.density_of_states is not None:
            fermi_energy = self.get_self_consistent_fermi_energy()

            # Now that we have a new Fermi energy, we should uncache the
            # old number of electrons.
            if self._is_property_cached("number_of_electrons"):
                del self.number_of_electrons
            return fermi_energy

        else:
            # The Fermi energy is set to None instead of creating an error.
            # This is because the Fermi energy is used a lot throughout
            # and it may be benificial not to throw an error directly
            # but rather we need to revert to the DFT values.
            return None

    @cached_property
    def entropy_contribution(self):
        """Entropy contribution to the energy calculated via cached DOS."""
        if self.density_of_states is not None:
            return self.get_entropy_contribution()
        else:
            raise Exception(
                "No cached DOS available to calculate this property."
            )

    @cached_property
    def d_entropy_contribution_d_dos(self):
        """Derivative of band energy, calculated via cached DOS."""
        if self.density_of_states is not None:
            return self.get_d_entropy_contribution_d_dos()
        else:
            raise Exception("No cached DOS available to calculate this "
                            "property.")

    @cached_property
    def d_number_of_electrons_d_mu(self):
        """   """
        if self.density_of_states is not None:
            return self.get_d_number_of_electrons_d_mu()
        else:
            raise Exception("No cached DOS available to calculate this "
                            "property.")

    def uncache_properties(self):
        """Uncache all cached properties of this calculator."""
        if self._is_property_cached("number_of_electrons"):
            del self.number_of_electrons
        if self._is_property_cached("energy_grid"):
            del self.energy_grid
        if self._is_property_cached("band_energy"):
            del self.band_energy
        if self._is_property_cached("entropy_contribution"):
            del self.entropy_contribution
        if self._is_property_cached("fermi_energy"):
            del self.fermi_energy
        if self._is_property_cached("d_band_energy_dos"):
            del self.d_band_energy_d_dos
        if self._is_property_cached("d_entropy_contribution_d_dos"):
            del self.d_entropy_contribution_d_dos
        if self._is_property_cached("d_number_of_electrons_d_mu"):
            del self.d_number_of_electrons_d_mu

    ##############################
    # Methods
    ##############################

    # File I/O
    ##########

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert the units of an array into the MALA units.

        MALA units for the DOS means 1/eV.

        Parameters
        ----------
        array : numpy.ndarray
            Data for which the units should be converted.

        in_units : string
            Units of array. Currently supported are:

                 - 1/eV (no conversion, MALA unit)
                 - 1/Ry

        Returns
        -------
        converted_array : numpy.ndarray
            Data in 1/eV.
        """
        if in_units == "1/eV" or in_units is None:
            return array
        elif in_units == "1/Ry":
            return array * (1 / Rydberg)
        else:
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

        MALA units for the DOS means 1/eV.

        Parameters
        ----------
        array : numpy.ndarray
            Data in 1/eV.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.ndarray
            Data in out_units.
        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            raise Exception("Unsupported unit for LDOS.")

    def read_from_qe_dos_txt(self, path):
        """
        Read the DOS from a Quantum Espresso generated file.

        These files do not have a specified file ending, so I will call them
        qe.dos.txt here. QE saves the DOS in 1/eV.

        Parameters
        ----------
        path : string or List
            Path of the file containing the DOS.

        Returns
        -------
        dos_data:
            DOS data in 1/eV.
        """
        # Create the desired/specified energy grid. We will use this to
        # check whether we have a correct file.

        energy_grid = self.energy_grid

        if isinstance(path, str):
            readpaths = [path]
        else:
            readpaths = path

        current_energy_index = 0

        for path_index, readpath in enumerate(readpaths):
            return_dos_values = []

            # Open the file, then iterate through its contents.
            with open(readpath, "r") as infile:
                lines = infile.readlines()

                # Directly at the split we discard the last energy value
                # of the left side of the split. This requires that both
                # DOS have been sampled to/from the EXACT same value.
                # Currently, this responsibility lies with the user, and I
                # am not sure if we can consistently check for this, even
                # if we wanted to. In the DOS case, the energies get reported,
                # but that is NOT the case in the LDOS case.
                end = (
                    self.parameters.ldos_gridsize[path_index] - 1
                    if path_index != len(readpaths) - 1
                    else self.parameters.ldos_gridsize[path_index]
                )
                end += current_energy_index

                for dos_line in lines:
                    # The first column contains the energy value.
                    if "#" not in dos_line and current_energy_index < end:
                        e_val = float(dos_line.split()[0])
                        dosval = float(dos_line.split()[1])
                        if (
                            np.abs(e_val - energy_grid[current_energy_index])
                            < self.parameters.ldos_gridspacing_ev[path_index]
                            * 0.98
                        ):
                            return_dos_values.append(dosval)
                            current_energy_index += 1
                        # print(path_index, i)
            if path_index == 0:
                array = np.array(return_dos_values)
            else:
                array = np.concatenate((array, return_dos_values))
        self.density_of_states = array
        return array

    def read_from_qe_out(self, path=None, smearing_factor=2):
        r"""
        Calculate the DOS from a Quantum Espresso DFT output file.

        The DOS will be read calculated via the eigenvalues and the equation

        .. math:: D(E) = \sum_i \sum_k w_k \delta(\epsilon-\epsilon_{ik})

        Parameters
        ----------
        path : string
            Path to the QE out file. If None, the QE output that was loaded
            via read_additional_calculation_data will be used.

        smearing_factor : int or List
            Smearing factor relative to the energy grid spacing. Default is 2.
            If list, DOS splitting is assumed. In this case, the length
            of the list must be consistent with the length of
            parameters.targets.ldos_gridspacing_ev, which is a list in this
            case as well.

        Returns
        -------
        dos_data:
            DOS data in 1/eV.
        """
        # dos_per_band = delta_f(e_grid,dft.eigs)
        if path is None:
            atoms_object = self.atoms
        else:
            atoms_object = ase.io.read(path, format="espresso-out")
        kweights = atoms_object.get_calculator().get_k_point_weights()
        if kweights is None:
            raise Exception(
                "QE output file does not contain band information."
                "Rerun calculation with verbosity set to 'high'."
            )

        if isinstance(self.parameters.ldos_gridspacing_ev, list):
            grid_spacings = self.parameters.ldos_gridspacing_ev
            grid_sizes = self.parameters.ldos_gridsize
            _smearing_factors = smearing_factor
            if len(_smearing_factors) != len(grid_spacings):
                raise Exception(
                    "Length of smearing factor list must be consistent with "
                    "the length of the grid spacing list."
                )
        else:
            grid_spacings = [self.parameters.ldos_gridspacing_ev]
            grid_sizes = [self.parameters.ldos_gridsize]
            _smearing_factors = [smearing_factor]

        # Get the gaussians for all energy values and calculate the DOS per
        # band.
        dos_data = None
        previous_beginning = 0
        for spacing_idx, grid_spacing in enumerate(grid_spacings):
            size_for_spacing = grid_sizes[spacing_idx] + previous_beginning
            if spacing_idx != len(grid_spacings) - 1:
                size_for_spacing -= 1

            dos_per_band = gaussians(
                self.energy_grid[previous_beginning:size_for_spacing],
                atoms_object.get_calculator()
                .band_structure()
                .energies[0, :, :],
                _smearing_factors[spacing_idx] * grid_spacing,
            )
            dos_per_band = kweights[:, np.newaxis, np.newaxis] * dos_per_band

            # QE gives the band energies in eV, so no conversion necessary
            # here.
            if spacing_idx == 0:
                dos_data = np.sum(dos_per_band, axis=(0, 1))
            else:
                dos_data = np.concatenate(
                    (dos_data, np.sum(dos_per_band, axis=(0, 1)))
                )
            previous_beginning = size_for_spacing
        self.density_of_states = dos_data
        return dos_data

    def read_from_array(self, array, units="1/eV"):
        """
        Read the density data from a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            Numpy array containing the DOS.

        units : string
            Units the density is saved in. Usually none.
        """
        array *= self.convert_units(1, in_units=units)
        self.density_of_states = array
        return array

    def read_from_numpy_file(
        self, path, units=None, array=None, reshape=False
    ):
        """
        Read the data from a numpy file.

        Parameters
        ----------
        path : string
            Path to the numpy file.

        units : string
            Units the data is saved in.

        array : np.ndarray
            If not None, the array to save the data into.
            The array has to be 4-dimensional.

        Returns
        -------
        data : numpy.ndarray or None
            If array is None, a numpy array containing the data.
            Elsewise, None, as the data will be saved into the provided
            array.

        """
        loaded_array = np.load(path)
        self._process_loaded_array(loaded_array, units=units)
        return loaded_array

    # Calculations
    ##############

    def get_band_energy(
        self,
        dos_data=None,
        fermi_energy=None,
        temperature=None,
        integration_method="analytical",
        broadcast_band_energy=True,
    ):
        """
        Calculate the band energy from given DOS data.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data with dimension [energygrid]. If None, then the cached
            DOS will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        broadcast_band_energy : bool
            If True then the band energy will only be calculated on one
            rank and thereafter be distributed to all other ranks.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception(
                "No DOS data provided, cannot calculate this quantity."
            )

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout(
                    "Warning: No fermi energy was provided or could be "
                    "calculated from electronic structure data. "
                    "Using the DFT fermi energy, this may "
                    "yield unexpected results",
                    min_verbosity=1,
                )
                fermi_energy = self.fermi_energy_dft
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy
        if temperature is None:
            temperature = self.temperature

        if self.parameters._configuration["mpi"] and broadcast_band_energy:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                band_energy = self.__band_energy_from_dos(
                    dos_data,
                    energy_grid,
                    fermi_energy,
                    temperature,
                    integration_method,
                )
            else:
                band_energy = None

            band_energy = get_comm().bcast(band_energy, root=0)
            barrier()
            return band_energy
        else:
            energy_grid = self.energy_grid
            return self.__band_energy_from_dos(
                dos_data,
                energy_grid,
                fermi_energy,
                temperature,
                integration_method,
            )

    def get_d_band_energy_d_dos(self, dos_data=None, fermi_energy=None,
                                temperature=None, broadcast_derivative=True,
                                number_of_electrons=None):
        """
        Calculate the derivative of the band energy with respect to the DOS.

        Note that the derivative is taken at a constant number of electrons
        with the assumption that the change in the number of electrons is
        compensated by a change in the Fermi energy. This makes sense in
        metals, but probably does not make sense in insulators or
        semiconductors.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        broadcast_derivative : bool
            If True then the band energy will only be calculated on one
            rank and thereafter be distributed to all other ranks.

        Returns
        -------
        d_band_energy_d_DOS : numpy.array with dimension [energygrid].
            Derivative of the band energy (in eV) with respect to the DOS.
        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception("No DOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout("Warning: No fermi energy was provided or could be "
                         "calculated from electronic structure data. "
                         "Using the DFT fermi energy, this may "
                         "yield unexpected results", min_verbosity=1)
                fermi_energy = self.fermi_energy_dft
                number_of_electrons = self.number_of_electrons_exact
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy
            number_of_electrons = self.number_of_electrons

        if temperature is None:
            temperature = self.temperature

        if self.parameters._configuration["mpi"] and broadcast_derivative:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                d_band_energy_d_dos = self.\
                    __d_band_energy_d_dos_from_dos(dos_data, energy_grid,
                                                   fermi_energy, temperature,
                                                   number_of_electrons)
            else:
                d_band_energy_d_dos = None

            d_band_energy_d_dos = get_comm().bcast(d_band_energy_d_dos,
                                                   root=0)
            barrier()
            return d_band_energy_d_dos
        else:
            energy_grid = self.energy_grid
            return self.\
                __d_band_energy_d_dos_from_dos(dos_data, energy_grid,
                                               fermi_energy, temperature,
                                               number_of_electrons)

    def get_number_of_electrons(
        self,
        dos_data=None,
        fermi_energy=None,
        temperature=None,
        integration_method="analytical",
    ):
        """
        Calculate the number of electrons from given DOS data.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data with dimension [energygrid]. If None, then the cached
            DOS will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception(
                "No DOS data provided, cannot calculate this quantity."
            )

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout(
                    "Warning: No fermi energy was provided or could be "
                    "calculated from electronic structure data. "
                    "Using the DFT fermi energy, this may "
                    "yield unexpected results",
                    min_verbosity=1,
                )
                fermi_energy = self.fermi_energy_dft
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy

        if temperature is None:
            temperature = self.temperature
        energy_grid = self.energy_grid
        return self.__number_of_electrons_from_dos(
            dos_data,
            energy_grid,
            fermi_energy,
            temperature,
            integration_method,
        )

    def get_entropy_contribution(
        self,
        dos_data=None,
        fermi_energy=None,
        temperature=None,
        integration_method="analytical",
        broadcast_entropy=True,
    ):
        """
        Calculate the entropy contribution to the total energy.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data with dimension [energygrid]. If None, then the cached
            DOS will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        broadcast_entropy : bool
            If True then the entropy will only be calculated on one
            rank and thereafter be distributed to all other ranks.

        Returns
        -------
        entropy_contribution : float
            S/beta in eV.
        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception(
                "No DOS data provided, cannot calculate this quantity."
            )

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout(
                    "Warning: No fermi energy was provided or could be "
                    "calculated from electronic structure data. "
                    "Using the DFT fermi energy, this may "
                    "yield unexpected results",
                    min_verbosity=1,
                )
                fermi_energy = self.fermi_energy_dft
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy
        if temperature is None:
            temperature = self.temperature

        if self.parameters._configuration["mpi"] and broadcast_entropy:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                entropy = self.__entropy_contribution_from_dos(
                    dos_data,
                    energy_grid,
                    fermi_energy,
                    temperature,
                    integration_method,
                )
            else:
                entropy = None

            entropy = get_comm().bcast(entropy, root=0)
            barrier()
            return entropy
        else:
            energy_grid = self.energy_grid
            return self.__entropy_contribution_from_dos(
                dos_data,
                energy_grid,
                fermi_energy,
                temperature,
                integration_method,
            )

    def get_d_entropy_contribution_d_dos(self, dos_data=None,
                                         fermi_energy=None,
                                         temperature=None,
                                         broadcast_derivative=True):
        """
        Calculate the derivative of the entropy contribution w.r.t the DOS.

        Note that the derivative is taken at a constant number of electrons
        with the assumption that the change in the number of electrons is
        compensated by a change in the Fermi energy. This makes sense in
        metals, but probably does not make sense in insulators or
        semiconductors.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid]. If None, then the cached
            DOS will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        broadcast_derivative : bool
            If True then the derivative will only be calculated on one
            rank and thereafter be distributed to all other ranks.

        Returns
        -------
        entropy_contribution : float
            S/beta in eV.
        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception("No DOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout("Warning: No fermi energy was provided or could be "
                         "calculated from electronic structure data. "
                         "Using the DFT fermi energy, this may "
                         "yield unexpected results", min_verbosity=1)
                fermi_energy = self.fermi_energy_dft
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy
        if temperature is None:
            temperature = self.temperature

        if self.parameters._configuration["mpi"] and broadcast_derivative:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                entropy = self. \
                    __d_entropy_contribution_d_dos_from_dos(dos_data, energy_grid,
                                                            fermi_energy, temperature)
            else:
                entropy = None

            entropy = get_comm().bcast(entropy, root=0)
            barrier()
            return entropy
        else:
            energy_grid = self.energy_grid
            return self. \
                __d_entropy_contribution_d_dos_from_dos(dos_data, energy_grid,
                                                fermi_energy, temperature)

    def get_d_number_of_electrons_d_mu(self, dos_data=None,
                                       fermi_energy=None,
                                       temperature=None,
                                       broadcast_derivative=True,
                                       delta=None):
        """

        Returns
        -------

        """
        # Parse the parameters.
        if dos_data is None and self.density_of_states is None:
            raise Exception("No DOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # DOS, or calculate everything from scratch.
        if dos_data is not None:
            if fermi_energy is None:
                printout("Warning: No fermi energy was provided or could be "
                         "calculated from electronic structure data. "
                         "Using the DFT fermi energy, this may "
                         "yield unexpected results", min_verbosity=1)
                fermi_energy = self.fermi_energy_dft
        else:
            dos_data = self.density_of_states
            fermi_energy = self.fermi_energy
        if temperature is None:
            temperature = self.temperature

        if delta is None:
            delta = self.parameters.delta_forces

        if self.parameters._configuration["mpi"] and broadcast_derivative:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                derivative = self. \
                    __d_number_of_electrons_d_mu(dos_data, energy_grid,
                                                 fermi_energy, temperature,
                                                 delta)
            else:
                derivative = None

            derivative = get_comm().bcast(derivative, root=0)
            barrier()
            return derivative
        else:
            energy_grid = self.energy_grid
            return self. \
                __d_number_of_electrons_d_mu(dos_data, energy_grid,
                                                fermi_energy, temperature,
                                             delta)

    def get_self_consistent_fermi_energy(
        self,
        dos_data=None,
        temperature=None,
        integration_method="analytical",
        broadcast_fermi_energy=True,
    ):
        r"""
        Calculate the self-consistent Fermi energy.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data with dimension [energygrid]. If None, then the cached
            DOS will be used for the calculation.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        broadcast_fermi_energy : bool
            If True then the Fermi energy will only be calculated on one
            rank and thereafter be distributed to all other ranks.

        Returns
        -------
        fermi_energy_self_consistent : float
            :math:`\epsilon_F` in eV.
        """
        if dos_data is None:
            dos_data = self.density_of_states
            if dos_data is None:
                raise Exception(
                    "No DOS data provided, cannot calculate this quantity."
                )

        if temperature is None:
            temperature = self.temperature

        if self.parameters._configuration["mpi"] and broadcast_fermi_energy:
            if get_rank() == 0:
                energy_grid = self.energy_grid
                fermi_energy_sc = toms748(
                    lambda fermi_sc: (
                        self.__number_of_electrons_from_dos(
                            dos_data,
                            energy_grid,
                            fermi_sc,
                            temperature,
                            integration_method,
                        )
                        - self.number_of_electrons_exact
                    ),
                    a=energy_grid[0],
                    b=energy_grid[-1],
                )
            else:
                fermi_energy_sc = None

            fermi_energy_sc = get_comm().bcast(fermi_energy_sc, root=0)
            barrier()
            return fermi_energy_sc
        else:
            energy_grid = self.energy_grid
            fermi_energy_sc = toms748(
                lambda fermi_sc: (
                    self.__number_of_electrons_from_dos(
                        dos_data,
                        energy_grid,
                        fermi_sc,
                        temperature,
                        integration_method,
                    )
                    - self.number_of_electrons_exact
                ),
                a=energy_grid[0],
                b=energy_grid[-1],
            )
            return fermi_energy_sc

    def get_density_of_states(self, dos_data=None):
        """
        Get the density of states.

        This function currently doesn't do much. In the LDOS and
        density equivalents of it, certain dimensionality reorderings
        may happen, this function purely exists for consistency
        reasons. In the future, that may change.
        """
        if dos_data is None:
            dos_data = self.density_of_states

        return dos_data

    # Private methods
    #################

    def _process_loaded_array(self, array, units=None):
        """
        Process loaded array (i.e., unit change, reshaping, etc.).

        Saves array to internal variable if class attribute save_target_data
        is True.

        Parameters
        ----------
        array : numpy.ndarray
            Array to process.

        units : string
            Units of input array.
        """
        array *= self.convert_units(1, in_units=units)
        if self.save_target_data:
            self.density_of_states = array

    def _set_feature_size_from_array(self, array):
        """
        Set the feature size from the array.

        Feature sizes are saved in different ways for different physical data
        classes.

        Parameters
        ----------
        array : numpy.ndarray
            Array to extract the feature size from.
        """
        self.parameters.ldos_gridsize = np.shape(array)[-1]

    @staticmethod
    def __number_of_electrons_from_dos(
        dos_data, energy_grid, fermi_energy, temperature, integration_method
    ):
        """
        Calculate the number of electrons from DOS data.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data (1D).

        energy_grid : numpy.ndarray
            Energy grid on which the DOS is defined.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        # Calculate the energy levels and the Fermi function.

        fermi_vals = fermi_function(
            energy_grid, fermi_energy, temperature, suppress_overflow=True
        )
        # Calculate the number of electrons.
        if integration_method == "trapezoid":
            number_of_electrons = integrate.trapezoid(
                dos_data * fermi_vals, energy_grid, axis=-1
            )
        elif integration_method == "simpson":
            number_of_electrons = integrate.simpson(
                dos_data * fermi_vals, energy_grid, axis=-1
            )
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            number_of_electrons, abserr = integrate.quad(
                lambda e: dos_pointer(e)
                * fermi_function(
                    e, fermi_energy, temperature, suppress_overflow=True
                ),
                energy_grid[0],
                energy_grid[-1],
                limit=500,
                points=fermi_energy,
            )
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(
                dos_data, "F0", "F1", fermi_energy, energy_grid, temperature
            )
        else:
            raise Exception("Unknown integration method.")

        return number_of_electrons

    @staticmethod
    def __band_energy_from_dos(
        dos_data, energy_grid, fermi_energy, temperature, integration_method
    ):
        """
        Calculate the band energy from DOS data.

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data (1D).

        energy_grid : numpy.ndarray
            Energy grid on which the DOS is defined.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        # Calculate the energy levels and the Fermi function.
        fermi_vals = fermi_function(
            energy_grid, fermi_energy, temperature, suppress_overflow=True
        )

        # Calculate the band energy.
        if integration_method == "trapz":
            band_energy = integrate.trapezoid(
                dos_data * (energy_grid * fermi_vals), energy_grid, axis=-1
            )
        elif integration_method == "simpson":
            band_energy = integrate.simpson(
                dos_data * (energy_grid * fermi_vals), energy_grid, axis=-1
            )
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            band_energy, abserr = integrate.quad(
                lambda e: dos_pointer(e)
                * e
                * fermi_function(
                    e, fermi_energy, temperature, suppress_overflow=True
                ),
                energy_grid[0],
                energy_grid[-1],
                limit=500,
                points=fermi_energy,
            )
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(
                dos_data, "F0", "F1", fermi_energy, energy_grid, temperature
            )
            band_energy_minus_uN = analytical_integration(
                dos_data, "F1", "F2", fermi_energy, energy_grid, temperature
            )
            band_energy = (
                band_energy_minus_uN + fermi_energy * number_of_electrons
            )
        else:
            raise Exception("Unknown integration method.")

        return band_energy

    @staticmethod
    def __d_band_energy_d_dos_from_dos(dos_data, energy_grid, fermi_energy,
                                       temperature, number_of_electrons):
        """Calculate the derivative of the band energy w.r.t. the DOS."""
        d_band_energy_minus_uN_d_dos = \
            analytical_integration_weights("F1", "F2", fermi_energy,
                                           energy_grid, temperature)
        d_number_of_electrons_d_dos = \
            analytical_integration_weights("F0", "F1", fermi_energy,
                                           energy_grid, temperature)

        # The following crude numerical derivatives could be replaced with
        # analytic derivatives with some work.Or they could be replaced with
        # more sophisticated numerical derivatives.
        delta = 0.001
        d_number_of_electrons_d_mu = (analytical_integration(dos_data, "F0",
                                                             "F1",
                                                             fermi_energy +
                                                             delta,
                                                             energy_grid,
                                                             temperature)
                                      - analytical_integration(dos_data, "F0",
                                                               "F1",
                                                               fermi_energy -
                                                               delta,
                                                               energy_grid,
                                                               temperature)) /\
                                     (2.0*delta)
        d_band_energy_minus_uN_d_mu = (analytical_integration(dos_data, "F1",
                                                              "F2",
                                                              fermi_energy +
                                                              delta,
                                                              energy_grid,
                                                              temperature)
                                       - analytical_integration(dos_data, "F1",
                                                                "F2",
                                                                fermi_energy
                                                                - delta,
                                                                energy_grid,
                                                                temperature))/\
                                      (2.0*delta)
        d_band_energy_d_mu = d_band_energy_minus_uN_d_mu + number_of_electrons

        # Since the number of electrons is constant
        # \frac{dN}{dDOS} = (\partial{dN}{dDOS})_{\mu} +
        # \frac{dN}{d\mu}*\frac{d\mu}{dDOS} = 0
        # So,
        #   \frac{d\mu}{dDOS} = - (\partial{dN}{dDOS})_{\mu} / \frac{dN}{d\mu}
        #
        # Also,
        #   \frac{d(E - \mu*N)}{dDOS} = (\partial{d(E - \mu*N)}{dDOS})_{\mu} +
        #   \frac{d(E - \mu*N)}{d\mu}*\frac{d\mu}{dDOS}
        #   \frac{dE}{dDOS} - \frac{d\mu}{dDOS}*N = (\partial{d(E -
        #   \mu*N)}{dDOS})_{\mu} + \frac{dE}{d\mu}*\frac{d\mu}{dDOS} -
        #   N*\frac{d\mu}{dDOS}
        #   \frac{dE}{dDOS} = (\partial{d(E - \mu*N)}{dDOS})_{\mu} +
        #   \frac{dE}{d\mu}*\frac{d\mu}{dDOS}
        #
        # Combining and rearranging,
        #   \frac{dE}{dDOS} = (\partial{d(E - \mu*N)}{dDOS})_{\mu} -
        #   \frac{dE}{d\mu}*(\partial{dN}{dDOS})_{\mu} / \frac{dN}{d\mu}
        #                   = (\partial{d(E - \mu*N)}{dDOS})_{\mu} -
        #                   (\partial{dN}{dDOS})_{\mu}*\frac{dE}{d\mu}/\frac{dN}{d\mu}

        d_band_energy_d_dos = d_band_energy_minus_uN_d_dos \
                              - d_number_of_electrons_d_dos * \
                              d_band_energy_d_mu / d_number_of_electrons_d_mu

        return d_band_energy_d_dos

    @staticmethod
    def __entropy_contribution_from_dos(
        dos_data, energy_grid, fermi_energy, temperature, integration_method
    ):
        r"""
        Calculate the entropy contribution to the total energy from DOS data.

        More specifically, this gives -\beta^-1*S_S

        Parameters
        ----------
        dos_data : numpy.ndarray
            DOS data (1D).

        energy_grid : numpy.ndarray
            Energy grid on which the DOS is defined.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapezoid" for trapezoid method
                - "simpson" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        entropy_contribution : float
            Entropy contribution in eV.
        """
        # Calculate the entropy contribution to the energy.
        if integration_method == "trapezoid":
            multiplicator = entropy_multiplicator(
                energy_grid, fermi_energy, temperature
            )
            entropy_contribution = integrate.trapezoid(
                dos_data * multiplicator, energy_grid, axis=-1
            )
            entropy_contribution /= get_beta(temperature)
        elif integration_method == "simpson":
            multiplicator = entropy_multiplicator(
                energy_grid, fermi_energy, temperature
            )
            entropy_contribution = integrate.simpson(
                dos_data * multiplicator, energy_grid, axis=-1
            )
            entropy_contribution /= get_beta(temperature)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_grid, dos_data)
            entropy_contribution, abserr = integrate.quad(
                lambda e: dos_pointer(e)
                * entropy_multiplicator(e, fermi_energy, temperature),
                energy_grid[0],
                energy_grid[-1],
                limit=500,
                points=fermi_energy,
            )
            entropy_contribution /= get_beta(temperature)
        elif integration_method == "analytical":
            entropy_contribution = analytical_integration(
                dos_data, "S0", "S1", fermi_energy, energy_grid, temperature
            )
        else:
            raise Exception("Unknown integration method.")

        return entropy_contribution


    @staticmethod
    def __d_entropy_contribution_d_dos_from_dos(dos_data, energy_grid, fermi_energy,
                                                temperature):
        """Calculate the derivative of the entropy  w.r.t the DOS."""
        d_entropy_contribution_d_dos = \
            analytical_integration_weights("S0", "S1", fermi_energy,
                                           energy_grid, temperature)
        d_number_of_electrons_d_dos = \
            analytical_integration_weights("F0", "F1", fermi_energy,
                                           energy_grid, temperature)
        # The following crude numerical derivatives could be replaced with
        # analytic derivatives with some work.
        # Or they could be replaced with more sophisticated numerical
        # derivatives.
        delta = 1e-10
        d_number_of_electrons_d_mu = (analytical_integration(dos_data, "F0", "F1",
                                                             fermi_energy + delta,
                                                             energy_grid,
                                                             temperature) \
                                      - analytical_integration(dos_data, "F0", "F1",
                                                               fermi_energy - delta,
                                                               energy_grid,
                                                               temperature)) / (
                                                 2.0 * delta)
        d_entropy_contribution_d_mu = (analytical_integration(dos_data, "S0", "S1",
                                                              fermi_energy + delta,
                                                              energy_grid,
                                                              temperature) \
                                       - analytical_integration(dos_data, "S0", "S1",
                                                                fermi_energy - delta,
                                                                energy_grid,
                                                                temperature)) / (
                                                  2.0 * delta)

        # Since the number of electrons is constant
        # \frac{dN}{dDOS} = (\partial{dN}{dDOS})_{\mu} +
        # \frac{dN}{d\mu}*\frac{d\mu}{dDOS} = 0
        # So,
        #   \frac{d\mu}{dDOS} = - (\partial{dN}{dDOS})_{\mu} /
        #                      \frac{dN}{d\mu}
        #
        # Also,
        #   \frac{dTS}{dDOS} = (\partial{dTS}{dDOS})_{\mu} +
        #                      \frac{dTS}{d\mu}*\frac{d\mu}{dDOS}
        #                    = (\partial{dTS}{dDOS})_{\mu} -
        #                    \frac{dTS}{d\mu}*(\partial{dN}{dDOS})_{\mu} / \frac{dN}{d\mu}
        #                    = (\partial{dTS}{dDOS})_{\mu} -
        #                    (\partial{dN}{dDOS})_{\mu} * \frac{dTS}{d\mu}/\frac{dN}{d\mu}

        d_entropy_contribution_d_dos = d_entropy_contribution_d_dos \
                                       - d_number_of_electrons_d_dos * d_entropy_contribution_d_mu / d_number_of_electrons_d_mu
        return d_entropy_contribution_d_dos

    @staticmethod
    def __d_number_of_electrons_d_mu(dos_data, energy_grid, fermi_energy,
                                     temperature, delta):
        d_number_of_electrons_d_mu = (analytical_integration(dos_data, "F0",
                                                             "F1",
                                                             fermi_energy +
                                                             delta,
                                                             energy_grid,
                                                             temperature)
                                      - analytical_integration(dos_data, "F0",
                                                               "F1",
                                                               fermi_energy -
                                                               delta,
                                                               energy_grid,
                                                               temperature)) /\
                                     (2.0*delta)
        return d_number_of_electrons_d_mu

