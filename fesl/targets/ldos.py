from .cube_parser import read_cube
from .target_base import TargetBase
from .calculation_helpers import fermi_function, integrate_values_on_grid
import numpy as np
import math


class LDOS(TargetBase):
    """Local density of states.
    Evaluation follow the outline and code provided in/by https://arxiv.org/abs/2010.04905.
    """
    def __init__(self, p):
        super(LDOS, self).__init__(p)
        self.target_length = self.parameters.ldos_gridsize

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


    def get_density(self, ldos_data, fermi_energy_ev, temperature_K, conserve_dimensions=False):
        """Calculates the electronic density, from given ldos_data.
        Input variables:
            - ldos_data - can either have the form
                    gridpoints x energygrid
                            or
                    gridx x gridy x gridz x energygrid.
            - fermi_energy_ev: Fermi energy for the system for which this LDOS data was calculated in eV
            - temperature_K: Temperature of system in K
            - conserve_dimensions: If true, gridx x gridy x gridz x energygrid data will be transformed into data
                of the same dimensions. If false, no such transformation will be done, and the result will be in
                the shape of gridpoints x energygrid.
        """

        ldos_data_shape = np.shape(ldos_data)
        if len(ldos_data_shape) == 2:
            # We have the LDOS as gridpoints x energygrid, so no further operation is necessary.
            pass
        elif len(ldos_data_shape) == 4:
            # We have the LDOS as gridx x gridy x gridz x energygrid, so some reshaping needs to be done.
            ldos_data = ldos_data.reshape([ldos_data_shape[0]*ldos_data_shape[1]*ldos_data_shape[2], ldos_data_shape[3]])
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

        density_values = integrate_values_on_grid(ldos_data * (energy_values * fermi_values), energy_values, axis=-1,
                                                  method="simps")

        if len(ldos_data_shape) == 4 and conserve_dimensions is True:
            ldos_data_shape = list(ldos_data_shape)
            ldos_data_shape[-1] = 1
            density_values = density_values.reshape(ldos_data_shape)

        return density_values

