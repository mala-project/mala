import scipy as sp
from fesl.targets.calculation_helpers import *

#  Test our implementation of the analytical integration of [1]

# Values used for the test.
energies = np.array([6.83, 7.11])
gridsize = 2
e_fermi = 7.0
temp = 298

# Calculate the analytical values.
beta = 1 / (kB * temp)
x = beta * (energies - e_fermi)
fi_0 = np.zeros(gridsize, dtype=np.float)
fi_1 = np.zeros(gridsize, dtype=np.float)
fi_2 = np.zeros(gridsize, dtype=np.float)
for i in range(0, gridsize):
    # Calculate beta and x
    fi_0[i] = get_f0_value(x[i], beta)
    fi_1[i] = get_f1_value(x[i], beta)
    fi_2[i] = get_f2_value(x[i], beta)
aint_0 = fi_0[1] - fi_0[0]
aint_1 = fi_1[1] - fi_1[0]
aint_2 = fi_2[1] - fi_2[0]

# Calculate the numerically approximated values.

qint_0, abserr = sp.integrate.quad(
    lambda e: fermi_function_eV(e, e_fermi, temp),
    energies[0], energies[-1])
qint_1, abserr = sp.integrate.quad(
    lambda e: (e - e_fermi) * fermi_function_eV(e, e_fermi, temp),
    energies[0], energies[-1])
qint_2, abserr = sp.integrate.quad(
    lambda e: (e - e_fermi) ** 2 * fermi_function_eV(e, e_fermi, temp),
    energies[0], energies[-1])

print("                                        F0       F1       F2")
print("Analytical value:   {0:12.4e}{1:12.4e}{2:12.4e}".format(aint_0, aint_1, aint_2))
print("Numerical value:   {0:12.4e}{1:12.4e}{2:12.4e}".format(qint_0, qint_1, qint_2))
print("Error:   {0:12.4e}{1:12.4e}{2:12.4e}".format(aint_0 - qint_0, aint_1 - qint_1, aint_2 - qint_2))
