
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
     '3.0': {'300K': 12.0},\
     '4.0': {'300K': 16.0},\
     '5.0': {'300K': 20.0},\
     '6.0': {'300K': 24.0},\
     '7.0': {'300K': 26.0},\
     '8.0': {'300K': 30.0},\
     '9.0': {'300K': 34.0}}

ldos_e_lvls = 128


def ldos_to_density(ldos_vals, temp, gcc):

    shape = ldos_vals.shape

#    print(shape)

    batch_size = shape[0]

    emin = ldos_e_min[gcc][temp]
    emax = ldos_e_max[gcc][temp]


    e_spacing = (emax - emin) / ldos_e_lvls
   
    if (e_spacing == 0):
        print("\n\nError! No Energy spacing on file.\n\n")
        exit(0)
   

#    energy_vals = np.arange(emin, emax, e_spacing)
#    dens_vals = np.empty(batch_size) 
#    for v in range(batch_size):
#        dens_sum = 0.0
#        for e in range(ldos_e_lvls - 1):
            # Trapezoid Rule for two points
#            dens_sum += e_spacing * \
#                    (ldos_batch[v, e + 1] + ldos_batch[v, e]) / 2

    # Trapezoid rule over a batch of LDOS's
    # SUM_0^128 (E_i+1 - E_i) * (LDOS(E_i) + LDOS(E_i+1)) / 2
    dens_vals = e_spacing / 2 * \
            sum(ldos_vals[:, 0:ldos_e_lvls-1] + \
            ldos_vals[:, 1:ldos_e_lvls])


#        dens_vals[v] = dens_sum
#    print(dens_vals)
#    print("E_spacing %s, %sgcc: %4.4f" % (temp, gcc, e_spacing))


    return dens_vals
