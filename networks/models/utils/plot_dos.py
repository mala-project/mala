
import numpy as np
import matplotlib.pyplot as plt
import ldos_calc


temp = '298K'
gcc = '2.699'

num_ldos_plots = 10

filename = "LDOS_Comparisons_%s_%sgcc.png" % (temp, gcc)

ldos_npy_path = '/ascldap/users/johelli/Code/mlmm/mlmm-ldrd-data/networks/training_data/ldos_data/%s/%sgcc/ldos_200x200x200grid_128elvls.npy' % (temp, gcc)

qe_dos_txt_path = '/ascldap/users/acangi/q-e_calcs/Al/datasets/vasp_econ_snapshots/%s/%sg/170726180545.0/100Ry_k333/Al.dos' % (temp, gcc)


print("Grabbing ldos from %s" % ldos_npy_path)
print("Grabbing true dos from %s" % qe_dos_txt_path)

# LDOS for temp, gcc
ldos = np.load(ldos_npy_path)

#dos = ldos_calc.ldos_to_dos(ldos, temp, gcc)
dos = ldos_calc.ldos_to_dos_simple(ldos, temp, gcc)



e_lvls = dos.shape[0]
nx = ldos.shape[0]

e_min = ldos_calc.get_e_min(temp, gcc)
e_max = ldos_calc.get_e_max(temp, gcc)

e_grid = np.linspace(e_min, e_max, e_lvls)

true_dos = np.loadtxt(qe_dos_txt_path, skiprows=1)


true_bandE = ldos_calc.dos_to_bandenergy(true_dos[:,1], temp, gcc)


de1 = true_dos[1,0] - true_dos[0,0]
de2 = true_dos[2,0] - true_dos[1,0]

deN = true_dos[111,0] - true_dos[110, 0]

print("\ndE1: %f dE2: %f dEN: %f" % (de1, de2, deN))

print("\nBAND ENERGY: %f" % (true_bandE))



fig, (ax0, ax1) = plt.subplots(2,1)

ax0.set_xlabel('Energy (eV)')
ax0.set_ylabel('DOS')

dos_handle, = ax0.plot(e_grid, dos, '-k')
truedos_handle, = ax0.plot(true_dos[:,0], true_dos[:,1], '-g')

ax0.legend(handles=[dos_handle, truedos_handle], labels=["DOS (LDOS Integrated)", "DOS (QE Output)"])

ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('LDOS')
for i in range(num_ldos_plots):
    r = np.random.randint(0,nx,3)
    ax1.plot(e_grid, ldos[r[0], r[1], r[2], :])

plt.savefig(filename)


print("\nFile %s generated. Success!\n" % filename)
