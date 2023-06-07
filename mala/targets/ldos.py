"""LDOS calculation class."""
from functools import cached_property

from ase.units import Rydberg, Bohr, J, m
import math
import numpy as np
from scipy import integrate

from mala.common.parallelizer import get_comm, printout, get_rank, get_size, \
    barrier
from mala.common.parameters import DEFAULT_NP_DATA_DTYPE
from mala.targets.cube_parser import read_cube
from mala.targets.xsf_parser import read_xsf
from mala.targets.target import Target
from mala.targets.calculation_helpers import fermi_function, \
    analytical_integration, integrate_values_on_spacing
from mala.targets.dos import DOS
from mala.targets.density import Density


class LDOS(Target):
    """Postprocessing / parsing functions for the local density of states.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this LDOS object.
    """

    ##############################
    # Constructors
    ##############################

    def __init__(self, params):
        super(LDOS, self).__init__(params)
        self.local_density_of_states = None

    @classmethod
    def from_numpy_file(cls, params, path, units="1/(eV*A^3)"):
        """
        Create an LDOS calculator from a numpy array saved in a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path : string
            Path to file that is being read.

        units : string
            Units the LDOS is saved in.

        Returns
        -------
        ldos_calculator : mala.targets.ldos.LDOS
            LDOS calculator object.
        """
        return_ldos_object = LDOS(params)
        return_ldos_object.read_from_numpy_file(path, units=units)
        return return_ldos_object

    @classmethod
    def from_numpy_array(cls, params, array, units="1/(eV*A^3)"):
        """
        Create an LDOS calculator from a numpy array in memory.

        By using this function rather then setting the local_density_of_states
        object directly, proper unit coversion is ensured.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        array : numpy.ndarray
            Path to file that is being read.

        units : string
            Units the LDOS is saved in.

        Returns
        -------
        ldos_calculator : mala.targets.ldos.LDOS
            LDOS calculator object.
        """
        return_ldos_object = LDOS(params)
        return_ldos_object.read_from_array(array, units=units)
        return return_ldos_object

    @classmethod
    def from_cube_file(cls, params, path_name_scheme, units="1/(eV*A^3)",
                       use_memmap=None):
        """
        Create an LDOS calculator from multiple cube files.

        The files have to be located in the same directory.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path_name_scheme : string
            Naming scheme for the LDOS .cube files. Every asterisk will be
            replaced with an appropriate number for the LDOS files. Before
            the file name, please make sure to include the proper file path.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS.
            If run in MPI parallel mode, such a file MUST be provided.
        """
        return_ldos_object = LDOS(params)
        return_ldos_object.read_from_cube(path_name_scheme, units=units,
                                          use_memmap=use_memmap)
        return return_ldos_object

    @classmethod
    def from_xsf_file(cls, params, path_name_scheme, units="1/(eV*A^3)",
                      use_memmap=None):
        """
        Create an LDOS calculator from multiple xsf files.

        The files have to be located in the same directory.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path_name_scheme : string
            Naming scheme for the LDOS .xsf files. Every asterisk will be
            replaced with an appropriate number for the LDOS files. Before
            the file name, please make sure to include the proper file path.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS.
            If run in MPI parallel mode, such a file MUST be provided.
        """
        return_ldos_object = LDOS(params)
        return_ldos_object.read_from_xsf(path_name_scheme, units=units,
                                         use_memmap=use_memmap)
        return return_ldos_object

    @classmethod
    def from_openpmd_file(cls, params, path):
        """
        Create an LDOS calculator from an OpenPMD file.

        Supports all OpenPMD supported file endings.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path : string
            Path to OpenPMD file.

        Returns
        -------
        ldos_calculator : mala.targets.ldos.LDOS
            LDOS calculator object.
        """
        return_ldos_object = LDOS(params)
        return_ldos_object.read_from_openpmd_file(path)
        return return_ldos_object

    ##############################
    # Properties
    ##############################

    @property
    def feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        return self.parameters.ldos_gridsize

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "LDOS"

    @property
    def si_unit_conversion(self):
        """
        Numeric value of the conversion from MALA (ASE) units to SI.

        Needed for OpenPMD interface.
        """
        return (m**3)*J

    @property
    def si_dimension(self):
        """Dictionary containing the SI unit dimensions in OpenPMD format."""
        import openpmd_api as io

        return {io.Unit_Dimension.M: -1, io.Unit_Dimension.L: -5,
                io.Unit_Dimension.T: 2}

    @property
    def local_density_of_states(self):
        """Local density of states as 1D or 3D array."""
        return self._local_density_of_states

    @local_density_of_states.setter
    def local_density_of_states(self, new_ldos):
        self._local_density_of_states = new_ldos
        # Setting a new LDOS means we have to uncache priorly cached
        # properties.
        self.uncache_properties()

    def get_target(self):
        """
        Get the target quantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        return self.local_density_of_states

    def invalidate_target(self):
        """
        Invalidates the saved target wuantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        self.local_density_of_states = None

    def uncache_properties(self):
        """Uncache all cached properties of this calculator."""
        if self._is_property_cached("number_of_electrons"):
            del self.number_of_electrons
        if self._is_property_cached("energy_grid"):
            del self.energy_grid
        if self._is_property_cached("total_energy"):
            del self.total_energy
        if self._is_property_cached("band_energy"):
            del self.band_energy
        if self._is_property_cached("fermi_energy"):
            del self.fermi_energy
        if self._is_property_cached("density"):
            del self.density
        if self._is_property_cached("density_of_states"):
            del self.density_of_states
        if self._is_property_cached("_density_calculator"):
            del self._density_calculator
        if self._is_property_cached("_density_of_states_calculator"):
            del self._density_of_states_calculator
        if self._is_property_cached("entropy_contribution"):
            del self.entropy_contribution

    @cached_property
    def energy_grid(self):
        """Energy grid on which the LDOS is expressed."""
        return self.get_energy_grid()

    @cached_property
    def total_energy(self):
        """Total energy of the system, calculated via cached LDOS."""
        if self.local_density_of_states is not None:
            return self.get_total_energy()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def band_energy(self):
        """Band energy of the system, calculated via cached LDOS."""
        if self.local_density_of_states is not None:
            return self.get_band_energy()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def entropy_contribution(self):
        """Band energy of the system, calculated via cached LDOS."""
        if self.local_density_of_states is not None:
            return self.get_entropy_contribution()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def number_of_electrons(self):
        """
        Number of electrons in the system, calculated via cached LDOS.

        Does not necessarily match up exactly with KS-DFT provided values,
        due to discretization errors.
        """
        if self.local_density_of_states is not None:
            return self.get_number_of_electrons()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def fermi_energy(self):
        """
        "Self-consistent" Fermi energy of the system.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated. Calculated via cached LDOS
        """
        if self.local_density_of_states is not None:
            fermi_energy = self. \
                get_self_consistent_fermi_energy()

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
    def density(self):
        """Electronic density as calculated by the cached LDOS."""
        if self.local_density_of_states is not None:
            return self.get_density()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def density_of_states(self):
        """DOS as calculated by the cached LDOS."""
        if self.local_density_of_states is not None:
            return self.get_density_of_states()
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def _density_calculator(self):
        if self.local_density_of_states is not None:
            return Density.from_ldos_calculator(self)
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    @cached_property
    def _density_of_states_calculator(self):
        if self.local_density_of_states is not None:
            return DOS.from_ldos_calculator(self)
        else:
            raise Exception("No cached LDOS available to calculate this "
                            "property.")

    ##############################
    # Methods
    ##############################

    # File I/O
    ##########

    @staticmethod
    def convert_units(array, in_units="1/(eV*A^3)"):
        """
        Convert the units of an array into the MALA units.

        MALA units for the LDOS means 1/eV.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array. Currently supported are:

                 - 1/(eV*A^3) (no conversion, MALA unit)
                 - 1/(eV*Bohr^3)
                 - 1/(Ry*Bohr^3)


        Returns
        -------
        converted_array : numpy.array
            Data in 1/(eV*A^3).
        """
        if in_units == "1/(eV*A^3)" or in_units is None:
            return array
        elif in_units == "1/(eV*Bohr^3)":
            return array * (1/Bohr) * (1/Bohr) * (1/Bohr)
        elif in_units == "1/(Ry*Bohr^3)":
            return array * (1/Rydberg) * (1/Bohr) * (1/Bohr) * (1/Bohr)
        else:
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

        MALA units for the LDOS means 1/(eV*A^3).

        Parameters
        ----------
        array : numpy.array
            Data in 1/eV.

        out_units : string
            Desired units of output array. Currently supported are:

                 - 1/(eV*A^3) (no conversion, MALA unit)
                 - 1/(eV*Bohr^3)
                 - 1/(Ry*Bohr^3)


        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "1/(eV*A^3)":
            return array
        elif out_units == "1/(eV*Bohr^3)":
            return array * Bohr * Bohr * Bohr
        elif out_units == "1/(Ry*Bohr^3)":
            return array * Rydberg * Bohr * Bohr * Bohr
        else:
            raise Exception("Unsupported unit for LDOS.")

    def read_from_cube(self, path_scheme, units="1/(eV*A^3)",
                       use_memmap=None, **kwargs):
        """
        Read the LDOS data from multiple cube files.

        The files have to be located in the same directory.

        Parameters
        ----------
        path_scheme : string
            Naming scheme for the LDOS .cube files. Every asterisk will be
            replaced with an appropriate number for the LDOS files. Before
            the file name, please make sure to include the proper file path.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS. Only has an effect in MPI parallel mode.
            Usage will reduce RAM footprint while SIGNIFICANTLY
            impacting disk usage and
        """
        # First determine how many digits the last file in the list of
        # LDOS.cube files
        # will have.
        # QuantumEspresso saves the pp.x cube files like this:
        # tmp.pp001ELEMENT_ldos.cube
        # tmp.pp002ELEMENT_ldos.cube
        # tmp.pp003ELEMENT_ldos.cube
        # ...
        # tmp.pp100ELEMENT_ldos.cube
        return self._read_from_qe_files(path_scheme, units,
                                        use_memmap, ".cube", **kwargs)

    def read_from_xsf(self, path_scheme, units="1/(eV*A^3)",
                      use_memmap=None, **kwargs):
        """
        Read the LDOS data from multiple .xsf files.

        The files have to be located in the same directory. .xsf
        files are used by XCrysDen.

        Parameters
        ----------
        path_scheme : string
            Naming scheme for the LDOS .xsf files. Every asterisk will be
            replaced with an appropriate number for the LDOS files. Before
            the file name, please make sure to include the proper file path.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS. Only has an effect in MPI parallel mode.
            Usage will reduce RAM footprint while SIGNIFICANTLY
            impacting disk usage and
        """
        return self._read_from_qe_files(path_scheme, units,
                                        use_memmap, ".xsf", **kwargs)

    def read_from_array(self, array, units="1/(eV*A^3)"):
        """
        Read the LDOS data from a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            Numpy array containing the density.

        units : string
            Units the LDOS is saved in.
        """
        array *= self.convert_units(1, in_units=units)
        self.local_density_of_states = array
        return array

    # Calculations
    ##############

    def get_energy_grid(self):
        """
        Get energy grid.

        Returns
        -------
        e_grid : numpy.array
            Energy grid on which the LDOS is defined.
        """
        emin = self.parameters.ldos_gridoffset_ev

        emax = self.parameters.ldos_gridoffset_ev + \
            self.parameters.ldos_gridsize * \
            self.parameters.ldos_gridspacing_ev
        grid_size = self.parameters.ldos_gridsize
        linspace_array = (np.linspace(emin, emax, grid_size, endpoint=False))
        return linspace_array

    def get_total_energy(self, ldos_data=None, dos_data=None,
                         density_data=None, fermi_energy=None,
                         temperature=None, voxel=None,
                         grid_integration_method="summation",
                         energy_integration_method="analytical",
                         atoms_Angstrom=None, qe_input_data=None,
                         qe_pseudopotentials=None, create_qe_file=True,
                         return_energy_contributions=False):
        """
        Calculate the total energy from LDOS or given DOS + density data.

        If neither LDOS nor DOS+Density data is provided, the cached LDOS will
        be attempted to be used for the calculation.



        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, dos_data and density_data
            cannot be None.

        dos_data : numpy.array
            DOS data, as [energygrid].

        density_data : numpy.array
            Density data, either as [gridsize] or [gridx,gridy,gridz].

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        voxel : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        grid_integration_method : str
            Integration method used to integrate the density on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        energy_integration_method : string
            Integration method to integrate the DOS. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. (recommended)

        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        create_qe_file : bool
            If True, a QE input file will be created by MALA during the
            calculation. This is the default, however there may be
            cases in which it makes sense for the user to provide a custom
            one.

        return_energy_contributions : bool
            If True, a dictionary of energy contributions will be provided
            alongside the total energy. The default is False.

        Returns
        -------
        total_energy : float
            Total energy of the system (in eV).

        """
        # Get relevant values from DFT calculation, if not otherwise specified.
        if voxel is None:
            voxel = self.voxel
        if fermi_energy is None:
            if ldos_data is None:
                fermi_energy = self.fermi_energy
            if fermi_energy is None:
                printout("Warning: No fermi energy was provided or could be "
                         "calculated from electronic structure data. "
                         "Using the DFT fermi energy, this may "
                         "yield unexpected results", min_verbosity=1)
                fermi_energy = self.fermi_energy_dft
        if temperature is None:
            temperature = self.temperature

        # Here we check whether we will use our internal, cached
        # LDOS, or calculate everything from scratch.
        if ldos_data is not None or (dos_data is not None
                                     and density_data is not None):

            # In this case we calculate everything from scratch,
            # because the user either provided LDOS data OR density +
            # DOS data.

            # Calculate DOS data if need be.
            if dos_data is None:
                dos_data = self.get_density_of_states(ldos_data,
                                                      voxel=
                                                      voxel,
                                                      integration_method=
                                                      grid_integration_method)

            # Calculate density data if need be.
            if density_data is None:
                density_data = self.get_density(ldos_data,
                                                fermi_energy=fermi_energy,
                                                integration_method=energy_integration_method)

            # Now we can create calculation objects to get the necessary
            # quantities.
            dos_calculator = DOS.from_ldos_calculator(self)
            density_calculator = Density.from_ldos_calculator(self)

            # With these calculator objects we can calculate all the necessary
            # quantities to construct the total energy.
            # (According to Eq. 9 in [1])
            # Band energy (kinetic energy)
            e_band = dos_calculator.get_band_energy(dos_data,
                                                    fermi_energy=fermi_energy,
                                                    temperature=temperature,
                                                    integration_method=energy_integration_method)

            # Smearing / Entropy contribution
            e_entropy_contribution = dos_calculator. \
                get_entropy_contribution(dos_data, fermi_energy=fermi_energy,
                                         temperature=temperature,
                                         integration_method=energy_integration_method)

            # Density based energy contributions (via QE)
            density_contributions \
                = density_calculator. \
                get_energy_contributions(density_data,
                                         qe_input_data=qe_input_data,
                                         atoms_Angstrom=atoms_Angstrom,
                                         qe_pseudopotentials=
                                         qe_pseudopotentials,
                                         create_file=create_qe_file)
        else:
            # In this case, we use cached propeties wherever possible.
            ldos_data = self.local_density_of_states
            if ldos_data is None:
                raise Exception("No input data provided to caculate "
                                "total energy. Provide EITHER LDOS"
                                " OR DOS and density.")

            # With these calculator objects we can calculate all the necessary
            # quantities to construct the total energy.
            # (According to Eq. 9 in [1])
            # Band energy (kinetic energy)
            e_band = self.band_energy

            # Smearing / Entropy contribution
            e_entropy_contribution = self.entropy_contribution

            # Density based energy contributions (via QE)
            density_contributions = self._density_calculator.\
                total_energy_contributions

        e_total = e_band + density_contributions["e_rho_times_v_hxc"] + \
            density_contributions["e_hartree"] + \
            density_contributions["e_xc"] + \
            density_contributions["e_ewald"] +\
            e_entropy_contribution
        if return_energy_contributions:
            energy_contribtuons = {"e_band": e_band,
                                   "e_rho_times_v_hxc":
                                       density_contributions["e_rho_times_v_hxc"],
                                   "e_hartree":
                                       density_contributions["e_hartree"],
                                   "e_xc":
                                       density_contributions["e_xc"],
                                   "e_ewald": density_contributions["e_ewald"],
                                   "e_entropy_contribution":
                                       e_entropy_contribution}
            return e_total, energy_contribtuons
        else:
            return e_total

    def get_band_energy(self, ldos_data=None, fermi_energy=None,
                        temperature=None, voxel=None,
                        grid_integration_method="summation",
                        energy_integration_method="analytical"):
        """
        Calculate the band energy from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, the cached LDOS
            will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        grid_integration_method : str
            Integration method used to integrate the LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        energy_integration_method : string
            Integration method to integrate the DOS. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. (recommended)

        voxel : ase.cell.Cell
            Voxel to be used for grid intergation.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        if ldos_data is None and self.local_density_of_states is None:
            raise Exception("No LDOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # LDOS, or calculate everything from scratch.
        if ldos_data is not None:
            # The band energy is calculated using the DOS.
            if voxel is None:
                voxel = self.voxel

            dos_data = self.get_density_of_states(ldos_data, voxel,
                                                  integration_method=
                                                  grid_integration_method)

            # Once we have the DOS, we can use a DOS object to calculate
            # the band energy.
            dos_calculator = DOS.from_ldos_calculator(self)
            return dos_calculator. \
                get_band_energy(dos_data, fermi_energy=fermi_energy,
                                temperature=temperature,
                                integration_method=energy_integration_method)
        else:
            return self._density_of_states_calculator.band_energy

    def get_entropy_contribution(self, ldos_data=None, fermi_energy=None,
                                 temperature=None, voxel=None,
                                 grid_integration_method="summation",
                                 energy_integration_method="analytical"):
        """
        Calculate the entropy contribution from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, the cached LDOS
            will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        grid_integration_method : str
            Integration method used to integrate the LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        energy_integration_method : string
            Integration method to integrate the DOS. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. (recommended)

        voxel : ase.cell.Cell
            Voxel to be used for grid intergation.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        if ldos_data is None and self.local_density_of_states is None:
            raise Exception("No LDOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # LDOS, or calculate everything from scratch.
        if ldos_data is not None:
            # The band energy is calculated using the DOS.
            if voxel is None:
                voxel = self.voxel

            dos_data = self.get_density_of_states(ldos_data, voxel,
                                                  integration_method=
                                                  grid_integration_method)

            # Once we have the DOS, we can use a DOS object to calculate
            # the band energy.
            dos_calculator = DOS.from_ldos_calculator(self)
            return dos_calculator. \
                get_entropy_contribution(dos_data, fermi_energy=fermi_energy,
                                         temperature=temperature,
                                         integration_method=energy_integration_method)
        else:
            return self._density_of_states_calculator.entropy_contribution

    def get_number_of_electrons(self, ldos_data=None, voxel=None,
                                fermi_energy=None, temperature=None,
                                grid_integration_method="summation",
                                energy_integration_method="analytical"):
        """
        Calculate the number of electrons from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, the cached LDOS
            will be used for the calculation.

        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        grid_integration_method : str
            Integration method used to integrate the LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        energy_integration_method : string
            Integration method to integrate the DOS. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. (recommended)

        voxel : ase.cell.Cell
            Voxel to be used for grid intergation.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        if ldos_data is None and self.local_density_of_states is None:
            raise Exception("No LDOS data provided, cannot calculate"
                            " this quantity.")

        # Here we check whether we will use our internal, cached
        # LDOS, or calculate everything from scratch.
        if ldos_data is not None:
            # The number of electrons is calculated using the DOS.
            if voxel is None:
                voxel = self.voxel
            dos_data = self.get_density_of_states(ldos_data, voxel,
                                                  integration_method=
                                                  grid_integration_method)

            # Once we have the DOS, we can use a DOS object to calculate the
            # number of electrons.
            dos_calculator = DOS.from_ldos_calculator(self)
            return dos_calculator. \
                get_number_of_electrons(dos_data, fermi_energy=fermi_energy,
                                        temperature=temperature,
                                        integration_method=energy_integration_method)
        else:
            return self._density_of_states_calculator.number_of_electrons

    def get_self_consistent_fermi_energy(self, ldos_data=None, voxel=None,
                                         temperature=None,
                                         grid_integration_method="summation",
                                         energy_integration_method="analytical"):
        r"""
        Calculate the self-consistent Fermi energy.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, the cached LDOS
            will be used for the calculation.

        temperature : float
            Temperature in K.

        grid_integration_method : str
            Integration method used to integrate the LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        energy_integration_method : string
            Integration method to integrate the DOS. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. (recommended)

        voxel : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        Returns
        -------
        fermi_energy_self_consistent : float
            :math:`\epsilon_F` in eV.
        """
        if ldos_data is None and self.local_density_of_states is None:
            raise Exception("No LDOS data provided, cannot calculate"
                            " this quantity.")

        if ldos_data is not None:
            # The Fermi energy is calculated using the DOS.
            if voxel is None:
                voxel = self.voxel
            dos_data = self.get_density_of_states(ldos_data, voxel,
                                                  integration_method=
                                                  grid_integration_method)

            # Once we have the DOS, we can use a DOS object to calculate the
            # number of electrons.
            dos_calculator = DOS.from_ldos_calculator(self)
            return dos_calculator. \
                get_self_consistent_fermi_energy(dos_data,
                                                 temperature=temperature,
                                                 integration_method=energy_integration_method)
        else:
            return self._density_of_states_calculator.fermi_energy

    def get_density(self, ldos_data=None, fermi_energy=None, temperature=None,
                    conserve_dimensions=False, integration_method="analytical",
                    gather_density=False):
        """
        Calculate the density from given LDOS data.

        Parameters
        ----------
        conserve_dimensions : bool
            If True, the density is returned in the same dimensions as
            the LDOS was entered. If False, the density is always given
            as [gridsize, 1]. If None, the cached LDOS
            will be used for the calculation.


        fermi_energy : float
            Fermi energy level in eV.

        temperature : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        integration_method : string
            Integration method to integrate LDOS on energygrid.
            Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        gather_density : bool
            Only important if MPI is used. If True, the density will be
            gathered on rank 0.
            Helpful when using multiple CPUs for descriptor calculations
            and only one for network pass.

        Returns
        -------
        density_data : numpy.array
            Density data, dimensions depend on conserve_dimensions and LDOS
            dimensions.

        """
        if fermi_energy is None:
            if ldos_data is None:
                fermi_energy = self.fermi_energy
            if fermi_energy is None:
                printout("Warning: No fermi energy was provided or could be "
                         "calculated from electronic structure data. "
                         "Using the DFT fermi energy, this may "
                         "yield unexpected results", min_verbosity=1)
                fermi_energy = self.fermi_energy_dft
        if temperature is None:
            temperature = self.temperature

        if ldos_data is None:
            ldos_data = self.local_density_of_states
            if ldos_data is None:
                raise Exception("No LDOS data provided, cannot calculate"
                                " this quantity.")

        ldos_data_shape = np.shape(ldos_data)
        if len(ldos_data_shape) == 2:
            # We have the LDOS as gridpoints x energygrid,
            # so no further operation is necessary.
            ldos_data_used = ldos_data
            pass
        elif len(ldos_data_shape) == 4:
            # We have the LDOS as (gridx, gridy, gridz, energygrid),
            # so some reshaping needs to be done.
            ldos_data_used = ldos_data.reshape(
                [ldos_data_shape[0] * ldos_data_shape[1] * ldos_data_shape[2],
                 ldos_data_shape[3]])
            # We now have the LDOS as gridpoints x energygrid.

        else:
            raise Exception("Invalid LDOS array shape.")

        # Build the energy grid and calculate the fermi function.
        energy_grid = self.get_energy_grid()
        fermi_values = fermi_function(energy_grid, fermi_energy, temperature,
                                      suppress_overflow=True)

        # Calculate the number of electrons.
        if integration_method == "trapz":
            density_values = integrate.trapz(ldos_data_used * fermi_values,
                                             energy_grid, axis=-1)
        elif integration_method == "simps":
            density_values = integrate.simps(ldos_data_used * fermi_values,
                                             energy_grid, axis=-1)
        elif integration_method == "analytical":
            density_values = analytical_integration(ldos_data_used, "F0", "F1",
                                                    fermi_energy, energy_grid,
                                                    temperature)
        else:
            raise Exception("Unknown integration method.")

        # Now we have the full density; We now need to collect it, in the
        # MPI case.
        if self.parameters._configuration["mpi"] and gather_density:
            density_values = np.reshape(density_values,
                                        [np.shape(density_values)[0], 1])
            density_values = np.concatenate((self.local_grid, density_values),
                                            axis=1)
            full_density = self._gather_density(density_values)
            if len(ldos_data_shape) == 2:
                ldos_shape = np.shape(full_density)
                full_density = np.reshape(full_density, [ldos_shape[0] *
                                                         ldos_shape[1] *
                                                         ldos_shape[2], 1])
            return full_density
        else:
            if len(ldos_data_shape) == 4 and conserve_dimensions is True:
                # This breaks in the distributed case currently,
                # but I don't see an application where we would need it.
                # It would mean that we load an LDOS in distributed 3D fashion,
                # which is not implemented either.
                ldos_data_shape = list(ldos_data_shape)
                ldos_data_shape[-1] = 1
                density_values = density_values.reshape(ldos_data_shape)
            else:
                if len(ldos_data_shape) == 4:
                    grid_length = ldos_data_shape[0] * ldos_data_shape[1] * \
                                  ldos_data_shape[2]
                else:
                    grid_length = ldos_data_shape[0]
                density_values = density_values.reshape([grid_length, 1])
            return density_values

    def get_density_of_states(self, ldos_data=None, voxel=None,
                              integration_method="summation",
                              gather_dos=True):
        """
        Calculate the density of states from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid]. If None, the cached LDOS
            will be used for the calculation.


        voxel : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        integration_method : str
            Integration method used to integrate LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        gather_dos : bool
            Only important if MPI is used. If True, the DOS will be
            are gathered on rank 0.
            Helpful when using multiple CPUs for descriptor calculations
            and only one for network pass.

        Returns
        -------
        dos_values : np.array
            The DOS.
        """
        if ldos_data is None:
            ldos_data = self.local_density_of_states
            if ldos_data is None:
                raise Exception("No LDOS data provided, cannot calculate"
                                " this quantity.")

        if voxel is None:
            voxel = self.voxel

        ldos_data_shape = np.shape(ldos_data)
        if len(ldos_data_shape) != 4:
            if len(ldos_data_shape) != 2:
                raise Exception("Unknown LDOS shape, cannot calculate DOS.")
            elif integration_method != "summation":
                raise Exception("If using a 2D LDOS array, you can only "
                                "use summation as integration method.")

        # We have the LDOS as (gridx, gridy, gridz, energygrid), no
        # further operation is necessary.
        dos_values = ldos_data  # .copy()

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not
        # integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.
        grid_spacing_x = np.linalg.norm(voxel[0])
        grid_spacing_y = np.linalg.norm(voxel[1])
        grid_spacing_z = np.linalg.norm(voxel[2])

        if integration_method != "summation":
            # X
            if ldos_data_shape[0] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_x,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[1],
                                                     ldos_data_shape[2],
                                                     ldos_data_shape[3]))
                dos_values *= grid_spacing_x

            # Y
            if ldos_data_shape[1] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_y,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[2],
                                                     ldos_data_shape[3]))
                dos_values *= grid_spacing_y

            # Z
            if ldos_data_shape[2] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_z,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, ldos_data_shape[3])
                dos_values *= grid_spacing_z
        else:
            if len(ldos_data_shape) == 4:
                dos_values = np.sum(ldos_data, axis=(0, 1, 2),
                                    dtype=np.float64) * \
                             voxel.volume
            if len(ldos_data_shape) == 2:
                dos_values = np.sum(ldos_data, axis=0,
                                    dtype=np.float64) * \
                             voxel.volume

        if self.parameters._configuration["mpi"] and gather_dos:
            # I think we should refrain from top-level MPI imports; the first
            # import triggers an MPI init, which can take quite long.
            from mpi4py import MPI

            comm = get_comm()
            comm.Barrier()
            dos_values_full = np.zeros_like(dos_values)
            comm.Reduce([dos_values, MPI.DOUBLE],
                        [dos_values_full, MPI.DOUBLE],
                        op=MPI.SUM, root=0)
            return dos_values_full
        else:
            return dos_values

    def get_atomic_forces(self, ldos_data, dE_dd, used_data_handler,
                          snapshot_number=0):
        r"""
        Get the atomic forces, currently work in progress.

        Will eventually give :math:`\frac{dE}{d \underline{\boldsymbol{R}}}`.
        Will currently only give :math:`\frac{dd}{dB}`.

        Parameters
        ----------
        ldos_data: torch.Tensor
            Scaled (!) torch tensor holding the LDOS data for the snapshot
            for which the atomic force should be calculated.

        dE_dd: np.array
            (WIP) Derivative of the total energy w.r.t the LDOS.
            Later on, this will be evaluated within this subroutine. For now
            it is provided from outside.

        used_data_handler: mala.data.data_handler.DataHandler
            DataHandler that was used to predict the LDOS for which the
            atomic forces are supposed to be calculated.

        snapshot_number:
            Snapshot number (number within the data handler) for which this
            LDOS prediction was performed. Always 0 in the inference case.

        Returns
        -------
        dd_dB: torch.tensor
            (WIP) Returns the scaled (!) derivative of the LDOS w.r.t to
            the descriptors.

        """
        # For now this only works with ML generated LDOS.
        # Gradient of the LDOS respect to the descriptors.
        ldos_data.backward(dE_dd)
        dd_dB = used_data_handler.get_test_input_gradient(snapshot_number)
        return dd_dB

    # Private methods
    #################

    def _process_loaded_array(self, array, units=None):
        array *= self.convert_units(1, in_units=units)
        if self.save_target_data:
            self.local_density_of_states = array

    def _set_feature_size_from_array(self, array):
        self.parameters.ldos_gridsize = np.shape(array)[-1]

    def _gather_density(self, density_values, use_pickled_comm=False):
        """
        Gathers the density on rank 0 and sorts it.

        Parameters
        ----------
        density_values : numpy.array
            Numpy array with the density slice of this ranks local grid.

        use_pickled_comm : bool
            If True, the pickled communication route from mpi4py is used.
            If False, a Recv/Sendv combination is used. I am not entirely
            sure what is faster. Technically Recv/Sendv should be faster,
            but I doubt my implementation is all that optimal. For the pickled
            route we can use gather(), which should be fairly quick.
            However, for large grids, one CANNOT use the pickled route;
            too large python objects will break it. Therefore, I am setting
            the Recv/Sendv route as default.
        """
        # Barrier to make sure all ranks have descriptors..
        comm = get_comm()
        barrier()

        # Gather the densities into a list.
        if use_pickled_comm:
            density_list = comm.gather(density_values, root=0)
        else:
            sendcounts = np.array(comm.gather(np.shape(density_values)[0],
                                              root=0))
            if get_rank() == 0:
                # print("sendcounts: {}, total: {}".format(sendcounts,
                #                                          sum(sendcounts)))

                # Preparing the list of buffers.
                density_list = []
                for i in range(0, get_size()):
                    density_list.append(np.empty(sendcounts[i]*4,
                                                 dtype=np.float64))
                # No MPI necessary for first rank. For all the others,
                # collect the buffers.
                density_list[0] = density_values
                for i in range(1, get_size()):
                    comm.Recv(density_list[i], source=i,
                              tag=100+i)
                    density_list[i] = \
                        np.reshape(density_list[i],
                                   (sendcounts[i], 4))
            else:
                comm.Send(density_values, dest=0, tag=get_rank()+100)
            barrier()
        # if get_rank() == 0:
        #     printout(np.shape(all_snap_descriptors_list[0]))
        #     printout(np.shape(all_snap_descriptors_list[1]))
        #     printout(np.shape(all_snap_descriptors_list[2]))
        #     printout(np.shape(all_snap_descriptors_list[3]))

        # Dummy for the other ranks.
        # (For now, might later simply broadcast to other ranks).
        full_density = np.zeros([1, 1, 1, 1])

        # Reorder the list.
        if get_rank() == 0:
            # Prepare the densities array.
            nx = self.grid_dimensions[0]
            ny = self.grid_dimensions[1]
            nz = self.grid_dimensions[2]
            full_density = np.zeros(
                [nx, ny, nz, 1])
            # Fill the full density array.
            for idx, local_density in enumerate(density_list):
                # We glue the individual cells back together, and transpose.
                first_x = int(local_density[0][0])
                first_y = int(local_density[0][1])
                first_z = int(local_density[0][2])
                last_x = int(local_density[-1][0])+1
                last_y = int(local_density[-1][1])+1
                last_z = int(local_density[-1][2])+1
                full_density[first_x:last_x,
                             first_y:last_y,
                             first_z:last_z] = \
                    np.reshape(local_density[:, 3],
                               [last_z-first_z, last_y-first_y,
                                last_x-first_x, 1]).transpose([2, 1, 0, 3])

        return full_density

    def _read_from_qe_files(self, path_scheme, units,
                            use_memmap, file_type, **kwargs):
        """
        Read the LDOS from QE produced files, i.e. one file per energy level.

        Can currently work with .xsf and .cube files.

        Parameters
        ----------
        path_scheme : string
            Naming scheme for the LDOS .xsf files. Every asterisk will be
            replaced with an appropriate number for the LDOS files. Before
            the file name, please make sure to include the proper file path.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS. Only has an effect in MPI parallel mode.
            Usage will reduce RAM footprint while SIGNIFICANTLY
            impacting disk usage and
        """
        use_fp64 = kwargs.get("use_fp64", False)
        return_local = kwargs.get("return_local", False)
        ldos_dtype = np.float64 if use_fp64 else DEFAULT_NP_DATA_DTYPE

        # Find out the number of digits that are needed to encode this
        # grid (by QE).
        digits = int(math.log10(self.parameters.ldos_gridsize)) + 1

        # Iterate over the amount of specified LDOS input files.
        # QE is a Fortran code, so everything is 1 based.
        printout("Reading "+str(self.parameters.ldos_gridsize) +
                 " LDOS files from"+path_scheme+".", min_verbosity=0)
        ldos_data = None
        if self.parameters._configuration["mpi"]:
            local_size = int(np.floor(self.parameters.ldos_gridsize /
                                      get_size()))
            start_index = get_rank()*local_size + 1
            if get_rank()+1 == get_size():
                local_size += self.parameters.ldos_gridsize % \
                                     get_size()
            end_index = start_index+local_size
        else:
            start_index = 1
            end_index = self.parameters.ldos_gridsize + 1
            local_size = self.parameters.ldos_gridsize

        for i in range(start_index, end_index):
            tmp_file_name = path_scheme
            tmp_file_name = tmp_file_name.replace("*", str(i).zfill(digits))

            # Open the cube file
            if file_type == ".cube":
                data, meta = read_cube(tmp_file_name)
            elif file_type == ".xsf":
                data, meta = read_xsf(tmp_file_name)
            else:
                raise Exception("Unknown QE data type.")

            # Once we have read the first cube file, we know the dimensions
            # of the LDOS and can prepare the array
            # in which we want to store the LDOS.
            if i == start_index:
                data_shape = np.shape(data)
                ldos_data = np.zeros((data_shape[0], data_shape[1],
                                      data_shape[2], local_size),
                                     dtype=ldos_dtype)

            # Convert and then append the LDOS data.
            data = data*self.convert_units(1, in_units=units)
            ldos_data[:, :, :, i-start_index] = data[:, :, :]
            self.grid_dimensions = list(np.shape(ldos_data)[0:3])

        # We have to gather the LDOS either file based or not.
        if self.parameters._configuration["mpi"]:
            barrier()
            data_shape = np.shape(ldos_data)
            if return_local:
                return ldos_data, start_index-1, end_index-1
            if use_memmap is not None:
                if get_rank() == 0:
                    ldos_data_full = np.memmap(use_memmap,
                                               shape=(data_shape[0],
                                                      data_shape[1],
                                                      data_shape[2],
                                                      self.parameters.
                                                      ldos_gridsize),
                                               mode="w+",
                                               dtype=ldos_dtype)
                barrier()
                if get_rank() != 0:
                    ldos_data_full = np.memmap(use_memmap,
                                               shape=(data_shape[0],
                                                      data_shape[1],
                                                      data_shape[2],
                                                      self.parameters.
                                                      ldos_gridsize),
                                               mode="r+",
                                               dtype=ldos_dtype)
                barrier()
                ldos_data_full[:, :, :, start_index-1:end_index-1] = \
                    ldos_data[:, :, :, :]
                self.local_density_of_states = ldos_data_full
                return ldos_data_full
            else:
                comm = get_comm()

                # First get the indices from all the ranks.
                indices = np.array(
                    comm.gather([get_rank(), start_index, end_index],
                                root=0))
                ldos_data_full = None
                if get_rank() == 0:
                    ldos_data_full = np.empty((data_shape[0], data_shape[1],
                                               data_shape[2], self.parameters.
                                               ldos_gridsize),dtype=ldos_dtype)
                    ldos_data_full[:, :, :, start_index-1:end_index-1] = \
                        ldos_data[:, :, :, :]

                    # No MPI necessary for first rank. For all the others,
                    # collect the buffers.
                    for i in range(1, get_size()):
                        local_start = indices[i][1]
                        local_end = indices[i][2]
                        local_size = local_end-local_start
                        ldos_local = np.empty(local_size*data_shape[0] *
                                              data_shape[1]*data_shape[2],
                                              dtype=ldos_dtype)
                        comm.Recv(ldos_local, source=i, tag=100 + i)
                        ldos_data_full[:, :, :, local_start-1:local_end-1] = \
                            np.reshape(ldos_local, (data_shape[0],
                                                    data_shape[1],
                                                    data_shape[2],
                                                    local_size))[:, :, :, :]
                else:
                    comm.Send(ldos_data, dest=0,
                              tag=get_rank() + 100)
                barrier()
                self.local_density_of_states = ldos_data_full
                return ldos_data_full
        else:
            self.local_density_of_states = ldos_data
            return ldos_data
