"""Electronic density calculation class."""
import os
import time

import ase.io
from ase.units import Rydberg, Bohr, m
from functools import cached_property
import numpy as np
try:
    import total_energy as te
except ModuleNotFoundError:
    pass

from mala.common.parallelizer import printout, parallel_warn, barrier, get_size
from mala.targets.target import Target
from mala.targets.calculation_helpers import integrate_values_on_spacing
from mala.targets.cube_parser import read_cube, write_cube
from mala.targets.calculation_helpers import integrate_values_on_spacing
from mala.targets.xsf_parser import read_xsf
from mala.targets.atomic_force import AtomicForce
from mala.descriptors.atomic_density import AtomicDensity


class Density(Target):
    """Postprocessing / parsing functions for the electronic density.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this Target object.
    """

    ##############################
    # Class attributes
    ##############################

    te_mutex = False

    ##############################
    # Constructors
    ##############################

    def __init__(self, params):
        super(Density, self).__init__(params)
        self.density = None

    @classmethod
    def from_numpy_file(cls, params, path, units="1/A^3"):
        """
        Create a Density calculator from a numpy array saved in a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path : string
            Path to file that is being read.

        units : string
            Units the density is saved in.

        Returns
        -------
        dens_object : mala.targets.density.Density
            Density calculator object.
        """
        return_density_object = Density(params)
        return_density_object.read_from_numpy_file(path, units=units)
        return return_density_object

    @classmethod
    def from_numpy_array(cls, params, array, units="1/A^3"):
        """
        Create a Density calculator from a numpy array in memory.

        By using this function rather then setting the density
        object directly, proper unit coversion is ensured.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        array : numpy.ndarray
            Path to file that is being read.

        units : string
            Units the density is saved in.

        Returns
        -------
        dens_object : mala.targets.density.Density
            Density calculator object.
        """
        return_dos = Density(params)
        return_dos.read_from_array(array, units=units)
        return return_dos

    @classmethod
    def from_cube_file(cls, params, path, units="1/A^3"):
        """
        Create a Density calculator from a cube file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        path : string
            Name of the cube file.

        units : string
            Units the density is saved in.

        Returns
        -------
        dens_object : mala.targets.density.Density
            Density object created from LDOS object.
        """
        return_density_object = Density(params)
        return_density_object.read_from_cube(path, units=units)
        return return_density_object

    @classmethod
    def from_xsf_file(cls, params, path, units="1/A^3"):
        """
        Create a Density calculator from a xsf file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this DOS object.

        path : string
            Name of the xsf file.

        units : string
            Units the density is saved in.

        Returns
        -------
        dens_object : mala.targets.density.Density
            Density object created from LDOS object.
        """
        return_density_object = Density(params)
        return_density_object.read_from_xsf(path, units=units)
        return return_density_object

    @classmethod
    def from_openpmd_file(cls, params, path):
        """
        Create an Density calculator from an OpenPMD file.

        Supports all OpenPMD supported file endings.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this LDOS object.

        path : string
            Path to OpenPMD file.

        Returns
        -------
        density_calculator : mala.targets.density.Density
            Density calculator object.
        """
        return_density_object = Density(params)
        return_density_object.read_from_openpmd_file(path)
        return return_density_object

    @classmethod
    def from_ldos_calculator(cls, ldos_object):
        """
        Create a Density calculator from an LDOS object.

        If the LDOS object has data associated with it, this data will
        be copied.

        Parameters
        ----------
        ldos_object : mala.targets.ldos.LDOS
            LDOS object used as input.

        Returns
        -------
        dens_object : mala.targets.density.Density
            Density object created from LDOS object.
        """
        return_density_object = Density(ldos_object.parameters)
        return_density_object.fermi_energy_dft = ldos_object.fermi_energy_dft
        return_density_object.temperature = ldos_object.temperature
        return_density_object.voxel = ldos_object.voxel
        return_density_object.number_of_electrons_exact = ldos_object.\
            number_of_electrons_exact
        return_density_object.band_energy_dft_calculation = ldos_object.\
            band_energy_dft_calculation
        return_density_object.grid_dimensions = ldos_object.grid_dimensions
        return_density_object.atoms = ldos_object.atoms
        return_density_object.qe_input_data = ldos_object.qe_input_data
        return_density_object.qe_pseudopotentials = ldos_object.\
            qe_pseudopotentials
        return_density_object.total_energy_dft_calculation = \
            ldos_object.total_energy_dft_calculation
        return_density_object.kpoints = ldos_object.kpoints
        return_density_object.number_of_electrons_from_eigenvals = \
            ldos_object.number_of_electrons_from_eigenvals
        return_density_object.local_grid = ldos_object.local_grid
        return_density_object._parameters_full = ldos_object._parameters_full
        return_density_object.y_planes = ldos_object.y_planes

        # If the source calculator has LDOS data, then this new object
        # can have DOS data.
        if ldos_object.local_density_of_states is not None:
            return_density_object.density = ldos_object.density

        return return_density_object

    ##############################
    # Properties
    ##############################

    @property
    def feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        return 1

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "Density"

    @property
    def si_unit_conversion(self):
        """
        Numeric value of the conversion from MALA (ASE) units to SI.

        Needed for OpenPMD interface.
        """
        return m**3

    @property
    def si_dimension(self):
        """Dictionary containing the SI unit dimensions in OpenPMD format."""
        import openpmd_api as io

        return {io.Unit_Dimension.L: -3}

    @property
    def density(self):
        """Electronic density."""
        return self._density

    @density.setter
    def density(self, new_density):
        self._density = new_density
        # Setting a new density means we have to uncache priorly cached
        # properties.
        self.uncache_properties()

    def get_target(self):
        """
        Get the target quantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        return self.density

    def invalidate_target(self):
        """
        Invalidates the saved target wuantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        self.density = None

    @cached_property
    def number_of_electrons(self):
        """
        Number of electrons in the system, calculated via cached Density.

        Does not necessarily match up exactly with KS-DFT provided values,
        due to discretization errors.
        """
        if self.density is not None:
            return self.get_number_of_electrons()
        else:
            raise Exception("No cached density available to "
                            "calculate this property.")

    @cached_property
    def total_energy_contributions(self):
        """
        All density based contributions to the total energy.

        Calculated via the cached density.
        """
        if self.density is not None:
            return self.get_energy_contributions()
        else:
            raise Exception("No cached density available to "
                            "calculate this property.")

    def uncache_properties(self):
        """Uncache all cached properties of this calculator."""
        if self._is_property_cached("number_of_electrons"):
            del self.number_of_electrons

        if self._is_property_cached("total_energy_contributions"):
            del self.total_energy_contributions

    ##############################
    # Methods
    ##############################

    # File I/O
    ##########

    @staticmethod
    def convert_units(array, in_units="1/A^3"):
        """
        Convert the units of an array into the MALA units.

        MALA units for the density means 1/A^3.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array. Currently, supported are:

                 - 1/A^3 (no conversion, MALA unit)
                 - 1/Bohr^3

        Returns
        -------
        converted_array : numpy.array
            Data in 1/A^3.
        """
        if in_units == "1/A^3" or in_units is None:
            return array
        elif in_units == "1/Bohr^3":
            return array * (1/Bohr) * (1/Bohr) * (1/Bohr)
        else:
            raise Exception("Unsupported unit for density.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

        MALA units for the density means 1/A^3.

        Parameters
        ----------
        array : numpy.array
            Data in 1/A^3.

        out_units : string
            Desired units of output array. Currently, supported are:

                 - 1/A^3 (no conversion, MALA unit)
                 - 1/Bohr^3

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "1/A^3":
            return array
        elif out_units == "1/Bohr^3":
            return array * Bohr * Bohr * Bohr
        else:
            raise Exception("Unsupported unit for density.")

    def read_from_cube(self, path, units="1/A^3", **kwargs):
        """
        Read the density data from a cube file.

        Parameters
        ----------
        path : string
            Name of the cube file.

        units : string
            Units the density is saved in. Usually none.
        """
        printout("Reading density from .cube file ", path, min_verbosity=0)
        data, meta = read_cube(path)
        data *= self.convert_units(1, in_units=units)
        self.density = data
        self.grid_dimensions = list(np.shape(data)[0:3])
        return data

    def read_from_xsf(self, path, units="1/A^3", **kwargs):
        """
        Read the density data from an xsf file.

        Parameters
        ----------
        path : string
            Name of the xsf file.

        units : string
            Units the density is saved in. Usually none.
        """
        printout("Reading density from .cube file ", path, min_verbosity=0)
        data, meta = read_xsf(path)*self.convert_units(1, in_units=units)
        self.density = data
        return data

    def read_from_array(self, array, units="1/A^3"):
        """
        Read the density data from a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            Numpy array containing the density.

        units : string
            Units the density is saved in. Usually none.
        """
        array *= self.convert_units(1, in_units=units)
        self.density = array
        return array

    def write_to_openpmd_file(self, path, array=None,
                              additional_attributes={},
                              internal_iteration_number=0):
        """
        Write data to a numpy file.

        Parameters
        ----------
        path : string
            File to save into. If no file ending is given, .h5 is assumed.

        array : numpy.ndarray
            Target data to save. If None, the data stored in the calculator
            will be used.

        additional_attributes : dict
            Dict containing additional attributes to be saved.

        internal_iteration_number : int
            Internal OpenPMD iteration number. Ideally, this number should
            match any number present in the file name, if this data is part
            of a larger data set.
        """
        if array is None:
            if len(self.density.shape) == 2:
                super(Target, self).\
                    write_to_openpmd_file(path, np.reshape(self.density,
                                                           self.grid_dimensions
                                                           + [1]),
                                          internal_iteration_number=
                                          internal_iteration_number)
            elif len(self.density.shape) == 4:
                super(Target, self).\
                    write_to_openpmd_file(path, self.density,
                                          internal_iteration_number=
                                          internal_iteration_number)
        else:
            super(Target, self).\
                write_to_openpmd_file(path, array,
                                      internal_iteration_number=
                                      internal_iteration_number)

    def write_to_cube(self, file_name, density_data=None, atoms=None,
                      grid_dimensions=None):
        """
        Write the density data in a cube file.

        Parameters
        ----------
        file_name : string
            Name of the file.

        density_data : numpy.ndarray
            1D or 3D array of the density.

        atoms : ase.Atoms
            Atoms to be written to the file alongside the density data.
            If None, and the target object has an atoms object, this will
            be used. Ignored, unless density_data is provided.

        grid_dimensions : list
            Grid dimensions. Ignored, unless density_data is provided.
        """
        if density_data is not None:
            if grid_dimensions is None or atoms is None:
                raise Exception("No grid or atom data provided. "
                                "Please note that these are only optional "
                                "if the density saved in the calculator is "
                                "used and have to be provided otherwise.")
        else:
            density_data = self.density
            grid_dimensions = self.grid_dimensions
            atoms = self.atoms

        if len(density_data.shape) == 4 or len(density_data.shape) == 2:
            density_data = np.reshape(density_data, grid_dimensions)
        else:
            raise Exception("Unknown density shape provided.")
        # %%
        meta = {}
        atom_list = []
        for i in range(0, len(atoms)):
            atom_list.append(
                (atoms[i].number, [4.0, ] + list(atoms[i].position / Bohr)))

        meta["atoms"] = atom_list
        meta["org"] = [0.0, 0.0, 0.0]
        meta["xvec"] = self.voxel[0] / Bohr
        meta["yvec"] = self.voxel[1] / Bohr
        meta["zvec"] = self.voxel[2] / Bohr
        write_cube(density_data, meta, file_name)

    # Calculations
    ##############

    def get_number_of_electrons(self, density_data=None, voxel=None,
                                integration_method="summation"):
        """
        Calculate the number of electrons from given density data.

        Parameters
        ----------
        density_data : numpy.array
            Electronic density on the given grid. Has to either be of the form
            gridpoints or (gridx, gridy, gridz). If None, then the cached
            density will be used for the calculation.


        voxel : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        integration_method : str
            Integration method used to integrate density on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)
        """
        if density_data is None:
            density_data = self.density
            if density_data is None:
                raise Exception("No density data provided, cannot calculate"
                                " this quantity.")

        if voxel is None:
            voxel = self.voxel

        # Check input data for correctness.
        data_shape = np.shape(density_data)
        if len(data_shape) != 4:
            if len(data_shape) != 2:
                raise Exception("Unknown Density shape, cannot calculate "
                                "number of electrons.")
            elif integration_method != "summation":
                raise Exception("If using a 1D density array, you can only"
                                " use summation as integration method.")

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not
        # integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        grid_spacing_bohr_x = np.linalg.norm(voxel[0])
        grid_spacing_bohr_y = np.linalg.norm(voxel[1])
        grid_spacing_bohr_z = np.linalg.norm(voxel[2])

        number_of_electrons = None
        if integration_method != "summation":
            number_of_electrons = density_data

            # X
            if data_shape[0] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr_x, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons =\
                    np.reshape(number_of_electrons, (data_shape[1],
                                                     data_shape[2]))
                number_of_electrons *= grid_spacing_bohr_x

            # Y
            if data_shape[1] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr_y, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons = \
                    np.reshape(number_of_electrons, (data_shape[2]))
                number_of_electrons *= grid_spacing_bohr_y

            # Z
            if data_shape[2] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr_z, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons *= grid_spacing_bohr_z
        else:
            if len(data_shape) == 4:
                number_of_electrons = np.sum(density_data, axis=(0, 1, 2)) \
                                      * voxel.volume
            if len(data_shape) == 2:
                number_of_electrons = np.sum(density_data, axis=0) * \
                                      voxel.volume

        return np.squeeze(number_of_electrons)

    def get_density(self, density_data=None, convert_to_threedimensional=False,
                    grid_dimensions=None):
        """
        Get the electronic density, based on density data.

        This function only does reshaping, no calculations.

        Parameters
        ----------
        density_data : numpy.array
            Electronic density data, this array will be returned unchanged
            depending on the other parameters. If None, then the cached
            density will be used for the calculation.

        convert_to_threedimensional : bool
            If True, then a density saved as a 1D array will be converted to
            a 3D array (gridsize -> gridx * gridy * gridz)

        grid_dimensions : list
            Provide a list of dimensions to be used in the transformation
            1D -> 3D. If None, MALA will attempt to use the values read with
            Target.read_additional_read_additional_calculation_data .
            If that cannot be done, this function will raise an exception.

        Returns
        -------
        density_data : numpy.array
            Electronic density data in the desired shape.
        """
        if len(density_data.shape) == 4:
            return density_data
        elif len(density_data.shape) == 2:
            if convert_to_threedimensional:
                if self.parameters._configuration["mpi"]:
                    # In the MPI case we have to use the local grid to
                    # reshape the density properly.

                    first_x = int(self.local_grid[0][0])
                    first_y = int(self.local_grid[0][1])
                    first_z = int(self.local_grid[0][2])
                    last_x = int(self.local_grid[-1][0]) + 1
                    last_y = int(self.local_grid[-1][1]) + 1
                    last_z = int(self.local_grid[-1][2]) + 1
                    # density_data_reshaped = np.zeros([last_x-first_x,
                    #                                   last_y-first_y,
                    #                                   last_z-first_z],
                    #                                  dtype=np.float64)
                    density_data = \
                        np.reshape(density_data,
                                   [last_z - first_z, last_y - first_y,
                                    last_x - first_x, 1]).transpose([2, 1, 0, 3])
                    return density_data
                else:
                    if grid_dimensions is None:
                        grid_dimensions = self.grid_dimensions
                    return density_data.reshape(grid_dimensions+[1])
            else:
                return density_data
        else:
            raise Exception("Unknown density data shape.")

    def get_energy_contributions(self, density_data=None, create_file=True,
                                 atoms_Angstrom=None, qe_input_data=None,
                                 qe_pseudopotentials=None):
        r"""
        Extract density based energy contributions from Quantum Espresso.

        Done via a Fortran module accesible through python using f2py.
        Returns: e_rho_times_v_hxc, e_hartree,  e_xc, e_ewald

        Parameters
        ----------
        density_data : numpy.array
            Density data on a grid. If None, then the cached
            density will be used for the calculation.

        create_file : bool
            If False, the last mala.pw.scf.in file will be used as input for
            Quantum Espresso. If True (recommended), MALA will create this
            file according to calculation parameters.

        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        Returns
        -------
        energies : dict
            A dict containing the following entries:

                - :math:`n\,V_\mathrm{xc}`
                - :math:`E_\mathrm{H}`
                - :math:`E_\mathrm{xc}`
                - :math:`E_\mathrm{Ewald}`
        """
        if density_data is None:
            density_data = self.density
            if density_data is None:
                raise Exception("No density data provided, cannot calculate"
                                " this quantity.")

        if atoms_Angstrom is None:
            atoms_Angstrom = self.atoms
        self.__setup_total_energy_module(density_data, atoms_Angstrom,
                                         create_file=create_file,
                                         qe_input_data=qe_input_data,
                                         qe_pseudopotentials=
                                         qe_pseudopotentials)

        # Get and return the energies.
        energies = np.array(te.get_energies())*Rydberg
        energies_dict = {"e_rho_times_v_hxc": energies[0],
                         "e_hartree": energies[1], "e_xc": energies[2],
                         "e_ewald": energies[3]}
        return energies_dict

    def get_atomic_forces(self, density_data=None, create_file=True,
                          atoms_Angstrom=None, qe_input_data=None,
                          qe_pseudopotentials=None):
        """
        Calculate the atomic forces.

        This function uses an interface to QE. The atomic forces are
        calculated via the Hellman-Feynman theorem, although only the local
        contributions are calculated. The non-local contributions, as well
        as the SCF correction (so anything wavefunction dependent) are ignored.
        Therefore, this function is best used for data that was created using
        local pseudopotentials.

        Parameters
        ----------
        density_data : numpy.array
            Density data on a grid. If None, then the cached
            density will be used for the calculation.

        create_file : bool
            If False, the last mala.pw.scf.in file will be used as input for
            Quantum Espresso. If True (recommended), MALA will create this
            file according to calculation parameters.

        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        Returns
        -------
        atomic_forces : numpy.ndarray
            An array of the form (natoms, 3), containing the atomic forces
            in eV/Ang.

        """
        if density_data is None:
            density_data = self.density
            if density_data is None:
                raise Exception("No density data provided, cannot calculate"
                                " this quantity.")

        # First, set up the total energy module for calculation.
        if atoms_Angstrom is None:
            atoms_Angstrom = self.atoms
        self.__setup_total_energy_module(density_data, atoms_Angstrom,
                                         create_file=create_file,
                                         qe_input_data=qe_input_data,
                                         qe_pseudopotentials=
                                         qe_pseudopotentials)

        # Now calculate the forces.
        atomic_forces = np.array(te.calc_forces(len(atoms_Angstrom))).transpose()

        # QE returns the forces in Ry/Bohr.
        atomic_forces = AtomicForce.convert_units(atomic_forces,
                                                  in_units="Ry/Bohr")
        return atomic_forces

    @staticmethod
    def get_scaled_positions_for_qe(atoms):
        """
        Get the positions correctly scaled for QE.

        QE (for ibrav=0) scales a little bit different then ASE would.
        ASE uses all provided cell parameters, while QE simply sets the
        first entry in the cell parameter matrix as reference and divides
        all positions by this value.

        Parameters
        ----------
        atoms : ase.Atoms
            The atom objects for which the scaled positions should be
            calculated.

        Returns
        -------
        scaled_positions : numpy.array
            The scaled positions.
        """
        principal_axis = atoms.get_cell()[0][0]
        scaled_positions = atoms.get_positions()/principal_axis
        return scaled_positions

    # Private methods
    #################

    def _process_loaded_array(self, array, units=None):
        array *= self.convert_units(1, in_units=units)
        if self.save_target_data:
            self.density = array

    def _set_feature_size_from_array(self, array):
        # Feature size is always 1 in this case, no need to do anything.
        pass

    def __setup_total_energy_module(self, density_data, atoms_Angstrom,
                                    create_file=True, qe_input_data=None,
                                    qe_pseudopotentials=None):
        if create_file:
            # If not otherwise specified, use values as read in.
            if qe_input_data is None:
                qe_input_data = self.qe_input_data
            if qe_pseudopotentials is None:
                qe_pseudopotentials = self.qe_pseudopotentials

            self.write_tem_input_file(atoms_Angstrom, qe_input_data,
                                      qe_pseudopotentials,
                                      self.grid_dimensions,
                                      self.kpoints)

        # initialize the total energy module.
        # FIXME: So far, the total energy module can only be initialized once.
        # This is ok when the only thing changing
        # are the atomic positions. But no other parameter can currently be
        # changed in between runs...
        # There should be some kind of de-initialization function that allows
        # for this.

        if Density.te_mutex is False:
            printout("MALA: Starting QuantumEspresso to get density-based"
                     " energy contributions.", min_verbosity=0)
            barrier()
            t0 = time.perf_counter()
            te.initialize(self.y_planes)
            barrier()
            t1 = time.perf_counter()
            printout("time used by total energy initialization: ", t1 - t0)

            Density.te_mutex = True
            printout("MALA: QuantumEspresso setup done.", min_verbosity=0)
        else:
            printout("MALA: QuantumEspresso is already running. Except for"
                     " the atomic positions, no new parameters will be used.",
                     min_verbosity=0)

        # Before we proceed, some sanity checks are necessary.
        # Is the calculation spinpolarized?
        nr_spin_channels = te.get_nspin()
        if nr_spin_channels != 1:
            raise Exception("Spin polarization is not yet implemented.")

        # If we got values through the ASE parser - is everything consistent?
        number_of_atoms = te.get_nat()
        if create_file is True:
            if number_of_atoms != atoms_Angstrom.get_global_number_of_atoms():
                raise Exception("Number of atoms is inconsistent between MALA "
                                "and Quantum Espresso.")

        # We need to find out if the grid dimensions are consistent.
        # That depends on the form of the density data we received.
        number_of_gridpoints = te.get_nnr()
        if len(density_data.shape) == 4:
            number_of_gridpoints_mala = density_data.shape[0] * \
                                        density_data.shape[1] * \
                                        density_data.shape[2]
        elif len(density_data.shape) == 2:
            number_of_gridpoints_mala = density_data.shape[0]
        else:
            raise Exception("Density data has wrong dimensions. ")

        # If MPI is enabled, we NEED z-splitting for this to work.
        if self._parameters_full.use_mpi and \
                not self._parameters_full.descriptors.use_z_splitting:
            raise Exception("Cannot calculate the total energy if "
                            "the real space grid was not split in "
                            "z-direction.")

        # Check if we need to test the grid points.
        # We skip the check only if z-splitting is enabled and unequal
        # z-splits are to be expected, and no
        # y-splitting is enabled (since y-splitting currently works
        # for equal z-splitting anyway).
        if self._parameters_full.use_mpi and \
           self._parameters_full.descriptors.use_y_splitting == 0 \
           and int(self.grid_dimensions[2] / get_size()) != \
                  (self.grid_dimensions[2] / get_size()):
            pass
        else:
            if number_of_gridpoints_mala != number_of_gridpoints:
                raise Exception("Grid is inconsistent between MALA and"
                                " Quantum Espresso")

        # Now we need to reshape the density.
        density_for_qe = None
        if len(density_data.shape) == 4:
            density_for_qe = np.reshape(density_data, [number_of_gridpoints,
                                                       1], order='F')
        elif len(density_data.shape) == 2:
            parallel_warn("Using 1D density to calculate the total energy"
                          " requires reshaping of this data. "
                          "This is unproblematic, as long as you provided t"
                          "he correct grid_dimensions.")
            density_for_qe = self.get_density(density_data,
                                              convert_to_threedimensional=True)

            density_for_qe = np.reshape(density_for_qe,
                                        [number_of_gridpoints_mala, 1],
                                        order='F')

            # If there is an inconsistency between MALA and QE (which
            # can only happen in the uneven z-splitting case at the moment)
            # we need to pad the density array.
            if density_for_qe.shape[0] < number_of_gridpoints:
                grid_diff = number_of_gridpoints - number_of_gridpoints_mala
                density_for_qe = np.pad(density_for_qe,
                                        pad_width=((0, grid_diff), (0, 0)))

        # QE has the density in 1/Bohr^3
        density_for_qe *= self.backconvert_units(1, "1/Bohr^3")

        # Reset the positions. Some calculations (such as the Ewald sum)
        # is directly performed here, so it is not enough to simply
        # instantiate the process with the file.
        positions_for_qe = self.get_scaled_positions_for_qe(atoms_Angstrom)

        if self._parameters_full.descriptors.\
                use_atomic_density_energy_formula:
            # Calculate the Gaussian descriptors for the calculation of the
            # structure factors.
            barrier()
            t0 = time.perf_counter()
            gaussian_descriptors = \
                self._get_gaussian_descriptors_for_structure_factors(
                    atoms_Angstrom, self.grid_dimensions)
            barrier()
            t1 = time.perf_counter()
            printout("time used by gaussian descriptors: ", t1 - t0,
                     min_verbosity=2)

            #
            # Check normalization of the Gaussian descriptors
            #
            # from mpi4py import MPI
            # ggrid_sum = np.sum(gaussian_descriptors)
            # full_ggrid_sum = np.array([0.0])
            # comm = get_comm()
            # comm.Barrier()
            # comm.Reduce([ggrid_sum, MPI.DOUBLE],
            # [full_ggrid_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
            # printout("full_ggrid_sum =", full_ggrid_sum)

            # Calculate the Gaussian descriptors for a reference system
            # consisting of one atom at position (0.0,0.0,0.0)
            barrier()
            t0 = time.perf_counter()
            atoms_reference = atoms_Angstrom.copy()
            del atoms_reference[1:]
            atoms_reference.set_positions([(0.0, 0.0, 0.0)])
            reference_gaussian_descriptors = \
                self._get_gaussian_descriptors_for_structure_factors(
                    atoms_reference, self.grid_dimensions)
            barrier()
            t1 = time.perf_counter()
            printout("time used by reference gaussian descriptors: ", t1 - t0,
                     min_verbosity=2)

            #
            # Check normalization of the reference Gaussian descriptors
            #
            # reference_ggrid_sum = np.sum(reference_gaussian_descriptors)
            # full_reference_ggrid_sum = np.array([0.0])
            # comm = get_comm()
            # comm.Barrier()
            # comm.Reduce([reference_ggrid_sum, MPI.DOUBLE],
            # [full_reference_ggrid_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
            # printout("full_reference_ggrid_sum =", full_reference_ggrid_sum)

        barrier()
        t0 = time.perf_counter()

        # If the Gaussian formula is used, both the calculation of the
        # Ewald energy and the structure factor can be skipped.
        te.set_positions(np.transpose(positions_for_qe), number_of_atoms,
                         self._parameters_full.descriptors. \
                         use_atomic_density_energy_formula,
                         self._parameters_full.descriptors. \
                         use_atomic_density_energy_formula)
        barrier()
        t1 = time.perf_counter()
        printout("time used by set_positions: ", t1 - t0,
                 min_verbosity=2)

        barrier()

        if self._parameters_full.descriptors.\
                use_atomic_density_energy_formula:
            t0 = time.perf_counter()
            gaussian_descriptors = \
                np.reshape(gaussian_descriptors,
                           [number_of_gridpoints, 1], order='F')
            reference_gaussian_descriptors = \
                np.reshape(reference_gaussian_descriptors,
                           [number_of_gridpoints, 1], order='F')
            sigma = self._parameters_full.descriptors.\
                atomic_density_sigma
            sigma = sigma / Bohr
            te.set_positions_gauss(self._parameters_full.verbosity,
                                   gaussian_descriptors,
                                   reference_gaussian_descriptors,
                                   sigma,
                                   number_of_gridpoints, 1)
            barrier()
            t1 = time.perf_counter()
            printout("time used by set_positions_gauss: ", t1 - t0,
                     min_verbosity=2)

        # Now we can set the new density.
        barrier()
        t0 = time.perf_counter()
        te.set_rho_of_r(density_for_qe, number_of_gridpoints, nr_spin_channels)
        barrier()
        t1 = time.perf_counter()
        printout("time used by set_rho_of_r: ", t1 - t0,
                 min_verbosity=2)

        return atoms_Angstrom

    def _get_gaussian_descriptors_for_structure_factors(self, atoms, grid):
        descriptor_calculator = AtomicDensity(self._parameters_full)
        kwargs = {"return_directly": True, "use_fp64": True}
        return descriptor_calculator.\
            calculate_from_atoms(atoms, grid, **kwargs)[:, 6:]
