
import torch
import numpy as np

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
fermi_energy = 7.770

# Boltzmann's constant
#k = 1.0
k = 8.617333262145e-5

# Conversion factor from Rydberg to eV
Ry2eV = 13.6056980659

# E min/max accessors
def get_e_min(temp, gcc):
    return ldos_e_min[gcc][temp]

def get_e_max(temp, gcc):
    return ldos_e_max[gcc][temp]

# Fermi-Dirac distribution
def fermi_function(energy, temp):
    return 1.0 / (1.0 + np.exp((energy - fermi_energy) / (k * temp))) 

# LDOS -> Local Density
def ldos_to_density(ldos_vals, temp, gcc):

    # LDOS should be shape (BATCH_SIZE, NUM_E_LVLS)

    batch_size = ldos_vals.shape[0]
    ldos_e_lvls = ldos_vals.shape[1]

    emin = ldos_e_min[gcc][temp]
    emax = ldos_e_max[gcc][temp]

    e_spacing = (emax - emin) / ldos_e_lvls
   
    # Remove 'K' and make float
    temp = float(temp[:-1])


#    if (e_spacing == 0):
#        print("\n\nError! No Energy spacing on file.\n\n")
#        exit(0)
   
    energy_vals = np.linspace(emin, emax, ldos_e_lvls)

#    dens_vals = np.empty(batch_size) 
#    for v in range(batch_size):
#        dens_sum = 0.0
#        for e in range(ldos_e_lvls - 1):
            # Trapezoid Rule for two points
#            dens_sum += e_spacing * \
#                    (ldos_batch[v, e + 1] + ldos_batch[v, e]) / 2

    # Trapezoid rule over a batch of LDOS's
    # SUM_0^128 (E_i+1 - E_i) * (LDOS(E_i) + LDOS(E_i+1)) / 2
#    dens_vals = e_spacing / 2 * \
#            sum(ldos_vals[:, 0:ldos_e_lvls-1] + \
#            ldos_vals[:, 1:ldos_e_lvls])

    
    # Simpson's 3/8 (4 pt rule)
    # SUM_0^128 3/8 * (E_i+1 - E_i) (f(E_i) + 3 * f(E_i+1) + 
    #                                3 * f(E_i+2) + f(E_i+3))
    # where f(E) = LDOS(E) * fermi_funct(E)

    wgt = (1.0/8.0) * np.array([3.0, 9.0, 9.0, 3.0])
    
    dens_vals = e_spacing * \
            sum(wgt[0] * ldos_vals[:, 0::4] * \
                fermi_function(energy_vals[0::4], temp) + \
                wgt[1] * ldos_vals[:, 1::4] * \
                fermi_function(energy_vals[1::4], temp) + \
                wgt[2] * ldos_vals[:, 2::4] * \
                fermi_function(energy_vals[2::4], temp) + \
                wgt[3] * ldos_vals[:, 3::4] *
                fermi_function(energy_vals[3::4], temp)) \


#        dens_vals[v] = dens_sum
#    print(dens_vals)
#    print("E_spacing %s, %sgcc: %4.4f" % (temp, gcc, e_spacing))

    return dens_vals


# LDOS -> DOS
def ldos_to_dos(all_ldos_vals, temp, gcc):

    # LDOS should be shape (NX, NY, NZ, NUM_E_LVLS)

    # Simpson's 3/8 rule (Closed) for integration over 3 cells
    # defined by 4 pts.
    # Ex:       |___|___|___|
    #           x0  x1  x2  x3
 
    # Closed (cell-edges)
    wgt = (1.0/8.0) * np.array([3.0, 9.0, 9.0, 3.0])

    # Netwon-Cotes Degree 5 rule (Open) for integration over 3 cells
    # defined by 4 pts.
    # Ex:       |___|___|___|
    #           x0  x1  x2  x3

    # Open (cell-centers)
#    wgt = (5.0/24.0) * np.array([11.0, 1.0, 1.0, 11.0])

    # Integrate Z
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[:,:,0::4,:] + \
            wgt[1] * all_ldos_vals[:,:,1::4,:] + \
            wgt[2] * all_ldos_vals[:,:,2::4,:] + \
            wgt[3] * all_ldos_vals[:,:,3::4,:]), 2)

    # Integrate Y
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[:,0::4,:] + \
            wgt[1] * all_ldos_vals[:,1::4,:] + \
            wgt[2] * all_ldos_vals[:,2::4,:] + \
            wgt[3] * all_ldos_vals[:,3::4,:]), 1)
            
    # Integrate X
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[0::4,:] + \
            wgt[1] * all_ldos_vals[1::4,:] + \
            wgt[2] * all_ldos_vals[2::4,:] + \
            wgt[3] * all_ldos_vals[3::4,:]), 0)

    return all_ldos_vals


def ldos_to_dos_simple(all_ldos_vals, temp, gcc):
    # LDOS should be shape (NX, NY, NZ, NUM_E_LVLS)

    # Simpson's rule for integration over 2 cells with periodic functions
    # defined by 3 pts.
    # Ex:       |___|___|
    #           x0  x1  x2
    
    wgt = np.array([1.0])

    # Integrate Z
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[:,:,:,:]), 2)
    
    # Integrate Y
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[:,:,:]), 1)
            
    # Integrate X
    all_ldos_vals = grid_spacing * np.sum(( \
            wgt[0] * all_ldos_vals[:,:]), 0)

    return all_ldos_vals


# DOS -> BANDENERGY
def dos_to_bandenergy(dos_vals, temp, gcc):
   
    ldos_e_lvls = dos_vals.shape[0]
    ldos_e_lvls = 136

    print("\nE_LVLS: %d" % ldos_e_lvls)


    dos_vals = dos_vals[0:ldos_e_lvls]

#    emin = ldos_e_min[gcc][temp]
#    emax = ldos_e_max[gcc][temp]

    emin = -4.627
    emax = 10.254 

    e_spacing = (emax - emin) / ldos_e_lvls
    energy_vals = np.linspace(emin, emax, ldos_e_lvls)
   
    # Remove 'K' and make float
    temp = float(temp[:-1])

    # Simpson's 3/8 (4 pt rule)
    # SUM_0^128 3/8 * (E_i+1 - E_i) (f(E_i) + 3 * f(E_i+1) + 
    #                                3 * f(E_i+2) + f(E_i+3))
    # where f(E) = E * LDOS(E) * fermi_funct(E)
    

    wgt = (1.0/8.0) * np.array([3.0, 9.0, 9.0, 3.0])
    
    band_energy = e_spacing * \
            sum(wgt[0] * energy_vals[0::4] * dos_vals[0::4] * \
                fermi_function(energy_vals[0::4], temp) + \
                wgt[1] * energy_vals[1::4] * dos_vals[1::4] * \
                fermi_function(energy_vals[1::4], temp) + \
                wgt[2] * energy_vals[2::4] * dos_vals[2::4] * \
                fermi_function(energy_vals[2::4], temp) + \
                wgt[3] * energy_vals[3::4] * dos_vals[3::4] *
                fermi_function(energy_vals[3::4], temp)) \

    return band_energy
    

# LDOS -> DOS -> BAND ENERGY
def ldos_to_bandenergy(all_ldos_vals, temp, gcc):

    return dos_to_bandenergy(ldos_to_dos(all_ldos_vals, temp, gcc), temp, gcc)




