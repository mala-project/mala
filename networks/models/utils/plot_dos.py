
import numpy as np
import matplotlib.pyplot as plt
import ldos_calc


temp = '298K'
gcc = '2.699'

num_ldos_plots = 10

filename = "LDOS_Comparisons_%s_%sgcc.png" % (temp, gcc)


# LDOS for temp, gcc
ldos = np.load('/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/networks/training_data/ldos_data/%s/%sgcc/ldos_200x200x200grid_128elvls.npy' % (temp, gcc))

dos = ldos_calc.ldos_to_dos(ldos, temp, gcc)


e_lvls = dos.shape[0]
nx = ldos.shape[0]

e_min = ldos_calc.get_e_min(temp, gcc)
e_max = ldos_calc.get_e_max(temp, gcc)

e_grid = np.linspace(e_min, e_max, e_lvls)

true_dos = np.loadtxt('/ascldap/users/acangi/q-e_calcs/Al/datasets/vasp_econ_snapshots/%s/%sg/170726180545.0/100Ry_k333/Al.dos' % (temp, gcc), skiprows=1)


fig, ax = plt.subplots(2,1)

ax[0].plot(e_grid, dos, '-k')
ax[0].plot(true_dos[:,0], true_dos[:,1], '-g')


for i in range(num_ldos_plots):
    r = np.random.randint(0,nx,3)
    ax[1].plot(e_grid, ldos[r[0], r[1], r[2], :])

plt.savefig(filename)

