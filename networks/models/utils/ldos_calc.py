
import torch
import numpy as np

import scipy.integrate

# Access with e_spacing['gcc']['temp']
ldos_e_min = \
    {'0.1': {'300K': 0.0},\
     '0.2': {'300K': 0.0},\
     '0.4': {'300K': -8.0},\
     '0.6': {'300K': -6.0},\
     '0.8': {'300K': -6.0},\
     '1.0': {'300K': -6.0},\
     '2.0': {'300K': -6.0},\
     '2.699': {'298K': -6.0},\
     '3.0': {'300K': -4.0},\
     '4.0': {'300K': -2.0},\
     '5.0': {'300K': -2.0},\
     '6.0': {'300K': 0.0},\
     '7.0': {'300K': 2.0},\
     '8.0': {'300K': 4.0},\
     '9.0': {'300K': 6.0}}

ldos_e_max = \
    {'0.1': {'300K': 0.0},\
     '0.2': {'300K': 0.0},\
     '0.4': {'300K': 0.0},\
     '0.6': {'300K': 0.0},\
     '0.8': {'300K': 2.0},\
     '1.0': {'300K': 2.0},\
     '2.0': {'300K': 8.0},\
     '2.699': {'298K': 12.0},\
     '3.0': {'300K': 12.0},\
     '4.0': {'300K': 16.0},\
     '5.0': {'300K': 20.0},\
     '6.0': {'300K': 24.0},\
     '7.0': {'300K': 26.0},\
     '8.0': {'300K': 30.0},\
     '9.0': {'300K': 34.0}}

#ldos_e_lvls = 128


# Parameters for 298K, 2.699gcc

# delta x/y/z
#grid_spacing = .08099
grid_spacing = .153049

# Fermi Level (2.699gcc/298K)
fermi_energy = 7.770345

# Boltzmann's constant
#k = 1.0
k = 8.617333262145e-5

# Conversion factor from Rydberg to eV
Ry2eV = 13.6056980659


### Accessors ###

# E min/max accessors
def get_e_min(temp, gcc):
    return ldos_e_min[gcc][temp]

def get_e_max(temp, gcc):
    return ldos_e_max[gcc][temp]

def get_Ry2eV():
    return Ry2eV

def get_fermi_energy():
    return fermi_energy

def get_grid_spacing():
    return grid_spacing

def get_k():
    return k

def get_energies(temp, gcc, e_lvls=128):
    
    e_min = get_e_min(temp, gcc)
    e_max = get_e_max(temp, gcc)
    
    return np.linspace(e_min, e_max, e_lvls)


### Functions ###

# Fermi-Dirac distribution
def fermi_function(energy, temp):
    return 1.0 / (1.0 + np.exp((energy - fermi_energy) / (k * temp))) 


# Integration wrappers for simpson's and trapezoid rule
# Simple True: Trapz, Simple False: Simps
def integrate_vals_grid(vals, x_grid, axis=0, simple=False):

    if (simple):
        return scipy.integrate.trapz(vals, x_grid, axis=axis)
    else:
        return scipy.integrate.simps(vals, x_grid, axis=axis)


def integrate_vals_spacing(vals, dx, axis=0, simple=False):
    
    if (simple):
        return scipy.integrate.trapz(vals, dx=dx, axis=axis)
    else:
        return scipy.integrate.simps(vals, dx=dx, axis=axis)


# LDOS -> Local Density
def ldos_to_density(ldos_vals, temp, gcc, simple=False):

    # LDOS should be shape (BATCH_SIZE, NUM_E_LVLS)
    # or shape (NX, NY, NZ, NUM_E_LVLS)

    ldos_shape = ldos_vals.shape

    if (len(ldos_vals.shape) == 2):
        print("Data in Gridpt/LDOS shape")
        
        batch_size = ldos_shape[0]

    
    elif (len(ldos_vals.shape) == 4):
        print("Data in X/Y/Z/LDOS shape")

        batch_size = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]
        np.reshape(ldos_vals, [batch_size, ldos_shape[-1]])
    
    else:
        print("Bad shape for ldos_vals. Use (Gridpts x LDOS) or (X x Y x Z x LDOS).")
        exit(0);


    ldos_e_lvls = ldos_shape[-1]
    emin = ldos_e_min[gcc][temp]
    emax = ldos_e_max[gcc][temp]

    e_spacing = (emax - emin) / ldos_e_lvls
   
    if (e_spacing == 0):
        print("\n\nError! No Energy spacing on file.\n\n")
        exit(0);
   
    # Remove 'K' and make float
    temp_val = float(temp[:-1])
    
    energy_vals = np.linspace(emin, emax, ldos_e_lvls)
    fermi_vals = fermi_function(energy_vals, temp_val)

    dens_vals = integrate_vals_grid(ldos_vals * (energy_vals * fermi_vals), energy_vals, axis=-1, simple=simple)

    if (len(ldos_shape) == 4):
        ldos_shape = list(ldos_shape)
        ldos_shape[-1] = 1
        np.reshape(dens_vals, ldos_shape)

    return dens_vals

# LDOS -> DOS
def ldos_to_dos(all_ldos_vals, temp, gcc, simple=False):

    # LDOS should be shape (NX, NY, NZ, NUM_E_LVLS)

    # Integrate X
    all_ldos_vals = integrate_vals_spacing(all_ldos_vals, grid_spacing, axis=0, simple=simple)

    # Integrate Y
    all_ldos_vals = integrate_vals_spacing(all_ldos_vals, grid_spacing, axis=0, simple=simple)

    # Integrate Z
    all_ldos_vals = integrate_vals_spacing(all_ldos_vals, grid_spacing, axis=0, simple=simple)

    return all_ldos_vals


# DOS -> BANDENERGY
def dos_to_band_energy(dos_vals, temp, gcc, simple=False):
   
    # DOS should be shape (NUM_E_LVLS)
    
    ldos_e_lvls = dos_vals.shape[0]

    emin = ldos_e_min[gcc][temp]
    emax = ldos_e_max[gcc][temp]

    e_spacing = (emax - emin) / ldos_e_lvls
    
    if (e_spacing == 0):
        print("\n\nError! No Energy spacing on file.\n\n")
        exit(0)
    
    # Remove 'K' and make float
    temp_val = float(temp[:-1])

    energy_vals = np.linspace(emin, emax, ldos_e_lvls)
    fermi_vals = fermi_function(energy_vals, temp_val)
    
    band_energy = integrate_vals_grid(dos_vals * (energy_vals * fermi_vals), energy_vals, axis=-1, simple=simple)

    return band_energy
    

# LDOS -> DOS -> BAND ENERGY
def ldos_to_band_energy(all_ldos_vals, temp, gcc, simple=False):

    # LDOS should be shape (NX, NY, NZ, NUM_E_LVLS)
   
    # LDOS -> DOS
    dos = ldos_to_dos(all_ldos_vals, temp, gcc, simple)

    # DOS -> BANDENERGY
    band_energy = dos_to_band_energy(dos, temp, gcc, simple)

    return band_energy




