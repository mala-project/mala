"""LDOS calculation class."""
from ase.units import Rydberg

import numpy as np
import math
import os
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass

from mala.common.parallelizer import get_comm, printout, get_rank, get_size, \
    barrier
from mala.targets.cube_parser import read_cube
from mala.targets.target import Target
from mala.targets.calculation_helpers import *
from mala.targets.dos import DOS
from mala.targets.density import Density


class LDOS(Target):
    """Postprocessing / parsing functions for the local density of states.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    def __init__(self, params):
        super(LDOS, self).__init__(params)
        self.target_length = self.parameters.ldos_gridsize
        self.cached_dos_exists = False
        self.cached_dos = []
        self.cached_density_exists = False
        self.cached_density = []

    def get_feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        return self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
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

        MALA units for the LDOS means 1/eV.

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

    def read_from_cube(self, file_name_scheme, directory, units="1/eV",
                       use_memmap=None):
        """
        Read the LDOS data from multiple cube files.

        The files have to be located in the same directory.

        Parameters
        ----------
        file_name_scheme : string
            Naming scheme for the LDOS .cube files.

        directory : string
            Directory containing the LDOS .cube files.

        units : string
            Units the LDOS is saved in.

        use_memmap : string
            If not None, a memory mapped file with this name will be used to
            gather the LDOS.
            If run in MPI parallel mode, such a file MUST be provided.

        Returns
        -------
        ldos_data : numpy.array
            Numpy array containing LDOS data.

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

        # Find out the number of digits that are needed to encode this
        # grid (by QE).
        digits = int(math.log10(self.parameters.ldos_gridsize)) + 1

        # Iterate over the amount of specified LDOS input files.
        # QE is a Fortran code, so everything is 1 based.
        printout("Reading "+str(self.parameters.ldos_gridsize) +
                 " LDOS files from"+directory+".", min_verbosity=0)
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
            tmp_file_name = file_name_scheme
            tmp_file_name = tmp_file_name.replace("*", str(i).zfill(digits))

            # Open the cube file
            data, meta = read_cube(os.path.join(directory, tmp_file_name))

            # Once we have read the first cube file, we know the dimensions
            # of the LDOS and can prepare the array
            # in which we want to store the LDOS.
            if i == start_index:
                data_shape = np.shape(data)
                ldos_data = np.zeros((data_shape[0], data_shape[1],
                                      data_shape[2], local_size),
                                     dtype=np.float64)

            # Convert and then append the LDOS data.
            data = data*self.convert_units(1, in_units=units)
            ldos_data[:, :, :, i-start_index] = data[:, :, :]

        # We have to gather the LDOS either file based or not.
        if self.parameters._configuration["mpi"]:
            barrier()
            data_shape = np.shape(ldos_data)
            if use_memmap is not None:
                if get_rank() == 0:
                    ldos_data_full = np.memmap(use_memmap,
                                               shape=(data_shape[0], data_shape[1],
                                                     data_shape[2], self.parameters.
                                                     ldos_gridsize), mode="w+",
                                               dtype=np.float64)
                barrier()
                if get_rank() != 0:
                    ldos_data_full = np.memmap(use_memmap,
                                               shape=(data_shape[0], data_shape[1],
                                                      data_shape[2], self.parameters.
                                                      ldos_gridsize), mode="r+",
                                               dtype=np.float64)
                barrier()
                ldos_data_full[:, :, :, start_index-1:end_index-1] = ldos_data[:, :, :, :]
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
                                               ldos_gridsize),dtype=np.float64)
                    ldos_data_full[:, :, :, start_index-1:end_index-1] = \
                        ldos_data[:, :, :, :]

                    # No MPI necessary for first rank. For all the others,
                    # collect the buffers.
                    for i in range(1, get_size()):
                        local_start = indices[i][1]
                        local_end = indices[i][2]
                        local_size = local_end-local_start
                        ldos_local = np.empty(local_size*data_shape[0]*data_shape[1]*data_shape[2], dtype=np.float64)
                        comm.Recv(ldos_local, source=i, tag=100 + i)
                        ldos_data_full[:, :, :, local_start-1:local_end-1] = np.reshape(ldos_local, (data_shape[0], data_shape[1], data_shape[2], local_size))[:,:,:,:]
                else:
                    comm.Send(ldos_data, dest=0,
                              tag=get_rank() + 100)
                barrier()
                return ldos_data_full
        else:
            return ldos_data

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
                         density_data=None, fermi_energy_eV=None,
                         temperature_K=None,
                         voxel_Bohr=None,
                         grid_integration_method="summation",
                         energy_integration_method="analytical",
                         atoms_Angstrom=None,
                         qe_input_data=None, qe_pseudopotentials=None,
                         create_qe_file=True,
                         return_energy_contributions=False):
        """
        Calculate the total energy from LDOS or given DOS + density data.

        EITHER LDOS OR Density+DOS data have to be specified. Elsewise
        this function will raise an exception.



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

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        voxel_Bohr : ase.cell.Cell
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
        if voxel_Bohr is None:
            voxel_Bohr = self.voxel_Bohr
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        # Check the input.
        if ldos_data is None:
            if dos_data is None or density_data is None:
                raise Exception("No input data provided to caculate "
                                "total energy. Provide EITHER LDOS"
                                " OR DOS and density.")

        # Calculate DOS data if need be.
        if dos_data is None:
            dos_data = self.get_density_of_states(ldos_data,
                                                  voxel_Bohr=
                                                  voxel_Bohr,
                                                  integration_method=
                                                  grid_integration_method)

        # Calculate density data if need be.
        if density_data is None:
            density_data = self.get_density(ldos_data,
                                            fermi_energy_ev=fermi_energy_eV,
                                            temperature_K=temperature_K,
                                            integration_method=
                                            energy_integration_method)

        # Now we can create calculation objects to get the necessary
        # quantities.
        dos_calculator = DOS.from_ldos(self)
        density_calculator = Density.from_ldos(self)

        # With these calculator objects we can calculate all the necessary
        # quantities to construct the total energy.
        # (According to Eq. 9 in [1])
        # Band energy (kinetic energy)
        e_band = dos_calculator.get_band_energy(dos_data,
                                                fermi_energy_eV=
                                                fermi_energy_eV,
                                                temperature_K=temperature_K,
                                                integration_method=
                                                energy_integration_method)

        # Smearing / Entropy contribution
        e_entropy_contribution = dos_calculator.\
            get_entropy_contribution(dos_data, fermi_energy_eV=fermi_energy_eV,
                                     temperature_K=temperature_K,
                                     integration_method=
                                     energy_integration_method)

        # Density based energy contributions (via QE)
        e_rho_times_v_hxc, e_hartree,  e_xc, e_ewald \
            = density_calculator.\
            get_energy_contributions(density_data, qe_input_data=qe_input_data,
                                     atoms_Angstrom=atoms_Angstrom,
                                     qe_pseudopotentials=qe_pseudopotentials,
                                     create_file=create_qe_file)
        e_total = e_band + e_rho_times_v_hxc + e_hartree + e_xc + e_ewald +\
            e_entropy_contribution
        if return_energy_contributions:
            energy_contribtuons = {"e_band": e_band,
                                   "e_rho_times_v_hxc": e_rho_times_v_hxc,
                                   "e_hartree": e_hartree,
                                   "e_xc": e_xc,
                                   "e_ewald": e_ewald,
                                   "e_entropy_contribution":
                                       e_entropy_contribution}
            return e_total, energy_contribtuons
        else:
            return e_total

    def get_band_energy(self, ldos_data, fermi_energy_eV=None,
                        temperature_K=None, voxel_Bohr=None,
                        grid_integration_method="summation",
                        energy_integration_method="analytical"):
        """
        Calculate the band energy from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
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

        voxel_Bohr : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        # The band energy is calculated using the DOS.
        if voxel_Bohr is None:
            voxel_Bohr = self.voxel_Bohr
        dos_data = self.get_density_of_states(ldos_data, voxel_Bohr,
                                              integration_method=
                                              grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate t
        # he band energy.
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.\
            get_band_energy(dos_data, fermi_energy_eV=fermi_energy_eV,
                            temperature_K=temperature_K,
                            integration_method=energy_integration_method)

    def get_number_of_electrons(self, ldos_data, voxel_Bohr=None,
                                fermi_energy_eV=None, temperature_K=None,
                                grid_integration_method="summation",
                                energy_integration_method="analytical"):
        """
        Calculate the number of electrons from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
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

        voxel_Bohr : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        # The number of electrons is calculated using the DOS.
        if voxel_Bohr is None:
            voxel_Bohr = self.voxel_Bohr
        dos_data = self.get_density_of_states(ldos_data, voxel_Bohr,
                                              integration_method=
                                              grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate the
        # number of electrons.
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.\
            get_number_of_electrons(dos_data, fermi_energy_eV=fermi_energy_eV,
                                    temperature_K=temperature_K,
                                    integration_method=
                                    energy_integration_method)

    def get_self_consistent_fermi_energy_ev(self, ldos_data,
                                            voxel_Bohr=None,
                                            temperature_K=None,
                                            grid_integration_method=
                                            "summation",
                                            energy_integration_method=
                                            "analytical"):
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
            [gridx,gridy,gridz,energygrid].

        temperature_K : float
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

        voxel_Bohr : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        Returns
        -------
        fermi_energy_self_consistent : float
            :math:`\epsilon_F` in eV.
        """
        # The Fermi energy is calculated using the DOS.
        if voxel_Bohr is None:
            voxel_Bohr = self.voxel_Bohr
        dos_data = self.get_density_of_states(ldos_data, voxel_Bohr,
                                              integration_method=
                                              grid_integration_method)

        # Once we have the DOS, we can use a DOS object to calculate the
        # number of electrons.
        dos_calculator = DOS.from_ldos(self)
        return dos_calculator.\
            get_self_consistent_fermi_energy_ev(dos_data,
                                                temperature_K=temperature_K,
                                                integration_method=
                                                energy_integration_method)

    def get_density(self, ldos_data, fermi_energy_ev=None, temperature_K=None,
                    conserve_dimensions=False,
                    integration_method="analytical",
                    gather_density=True):
        """
        Calculate the density from given LDOS data.

        Parameters
        ----------
        conserve_dimensions : bool
            If True, the density is returned in the same dimensions as
            the LDOS was entered. If False, the density is always given
            as [gridsize].

        fermi_energy_ev : float
            Fermi energy level in eV.

        temperature_K : float
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
        if self.cached_density_exists:
            return self.cached_density

        if fermi_energy_ev is None:
            fermi_energy_ev = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

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
        fermi_values = fermi_function(energy_grid, fermi_energy_ev,
                                      temperature_K, energy_units="eV")

        # Calculate the number of electrons.
        if integration_method == "trapz":
            density_values = integrate.trapz(ldos_data_used * fermi_values,
                                             energy_grid, axis=-1)
        elif integration_method == "simps":
            density_values = integrate.simps(ldos_data_used * fermi_values,
                                             energy_grid, axis=-1)
        elif integration_method == "analytical":
            density_values = analytical_integration(ldos_data_used, "F0", "F1",
                                                    fermi_energy_ev,
                                                    energy_grid,
                                                    temperature_K)
        else:
            raise Exception("Unknown integration method.")

        if len(ldos_data_shape) == 4 and conserve_dimensions is True:
            ldos_data_shape = list(ldos_data_shape)
            ldos_data_shape[-1] = 1
            density_values = density_values.reshape(ldos_data_shape)

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
            return density_values

    def get_density_of_states(self, ldos_data, voxel_Bohr=None,
                              integration_method="summation",
                              gather_dos=True):
        """
        Calculate the density of states from given LDOS data.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        voxel_Bohr : ase.cell.Cell
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
        if self.cached_dos_exists:
            return self.cached_dos

        if voxel_Bohr is None:
            voxel_Bohr = self.voxel_Bohr

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
        grid_spacing_bohr_x = np.linalg.norm(voxel_Bohr[0])
        grid_spacing_bohr_y = np.linalg.norm(voxel_Bohr[1])
        grid_spacing_bohr_z = np.linalg.norm(voxel_Bohr[2])

        if integration_method != "summation":
            # X
            if ldos_data_shape[0] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_bohr_x,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[1],
                                                     ldos_data_shape[2],
                                                     ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr_x

            # Y
            if ldos_data_shape[1] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_bohr_y,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, (ldos_data_shape[2],
                                                     ldos_data_shape[3]))
                dos_values *= grid_spacing_bohr_y

            # Z
            if ldos_data_shape[2] > 1:
                dos_values = integrate_values_on_spacing(dos_values,
                                                         grid_spacing_bohr_z,
                                                         axis=0,
                                                         method=
                                                         integration_method)
            else:
                dos_values = np.reshape(dos_values, ldos_data_shape[3])
                dos_values *= grid_spacing_bohr_z
        else:
            if len(ldos_data_shape) == 4:
                dos_values = np.sum(ldos_data, axis=(0, 1, 2)) * \
                             voxel_Bohr.volume
            if len(ldos_data_shape) == 2:
                dos_values = np.sum(ldos_data, axis=0) * \
                             voxel_Bohr.volume

        if self.parameters._configuration["mpi"] and gather_dos:
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
            the SNAP descriptors.

        """
        # For now this only works with ML generated LDOS.
        # Gradient of the LDOS respect to the SNAP descriptors.
        ldos_data.backward(dE_dd)
        dd_dB = used_data_handler.get_test_input_gradient(snapshot_number)
        return dd_dB

    def get_and_cache_density_of_states(self, ldos_data,
                                        voxel_Bohr=None,
                                        integration_method="summation"):
        """
        Calculate a DOS from LDOS data and keep it in memory.

        For all subsequent calculations involving the DOS, this cached
        DOS will be used. Usage of this function is advised for time-critical
        calculations.

        Parameters
        ----------
        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        voxel_Bohr : ase.cell.Cell
            Voxel to be used for grid intergation. Needs to reflect the
            symmetry of the simulation cell. In Bohr.

        integration_method : str
            Integration method used to integrate LDOS on the grid.
            Currently supported:

            - "trapz" for trapezoid method (only for cubic grids).
            - "simps" for Simpson method (only for cubic grids).
            - "summation" for summation and scaling of the values (recommended)

        Returns
        -------
        dos_values : np.array
            The DOS.

        """
        self.uncache_density_of_states()
        self.cached_dos = self.\
            get_density_of_states(ldos_data,
                                  voxel_Bohr=voxel_Bohr,
                                  integration_method=integration_method)
        self.cached_dos_exists = True
        return self.cached_dos

    def uncache_density_of_states(self):
        """Uncache a DOS, to calculate a new one in following steps."""
        self.cached_dos_exists = False

    def get_and_cache_density_cached(self, ldos_data,
                                     fermi_energy_ev=None,
                                     temperature_K=None,
                                     conserve_dimensions=False,
                                     integration_method="analytical"):
        """
        Calculate an electronic density from LDOS data and keep it in memory.

        For all subsequent calculations involving the electronic density, this
        cached density will be used. Usage of this function is advised for
        time-critical calculations.

        Parameters
        ----------
        conserve_dimensions : bool
            If True, the density is returned in the same dimensions as
            the LDOS was entered. If False, the density is always given
            as [gridsize].

        fermi_energy_ev : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        ldos_data : numpy.array
            LDOS data, either as [gridsize, energygrid] or
            [gridx,gridy,gridz,energygrid].

        integration_method : string
            Integration method to integrate LDOS on energygrid.
            Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        Returns
        -------
        density_data : numpy.array
            Density data, dimensions depend on conserve_dimensions and LDOS
            dimensions.
        """
        self.uncache_density()
        self.cached_density = self.\
            get_density(ldos_data,
                        fermi_energy_ev=fermi_energy_ev,
                        temperature_K=temperature_K,
                        conserve_dimensions=conserve_dimensions,
                        integration_method=integration_method)

        self.cached_density_exists = True
        return self.cached_density

    def uncache_density(self):
        """Uncache a density, to calculate a new one in following steps."""
        self.cached_density_exists = False

    def _gather_density(self, density_values, use_pickled_comm=False):
        """
        Gathers all SNAP descriptors on rank 0 and sorts them.

        This is useful for e.g. parallel preprocessing.
        This function removes the extra 3 components that come from parallel
        processing.
        I.e. if we have 91 SNAP descriptors, LAMMPS directly outputs us
        97 (in parallel mode), and this function returns, as to retain the
        3 x,y,z ones we by default include.

        Parameters
        ----------
        density_values : numpy.array
            Numpy array with the SNAP descriptors of this ranks local grid.

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

        # Gather the SNAP descriptors into a list.
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
            # Prepare the SNAP descriptor array.
            nx = self.grid_dimensions[0]
            ny = self.grid_dimensions[1]
            nz = self.grid_dimensions[2]
            full_density = np.zeros(
                [nx, ny, nz, 1])
            # Fill the full SNAP descriptors array.
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
                    np.reshape(local_density[:,3],[last_z-first_z,
                                                    last_y-first_y,
                                                    last_x-first_x,1]).transpose([2, 1, 0, 3])

        return full_density

