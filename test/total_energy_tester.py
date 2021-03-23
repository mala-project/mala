import total_energy as te
import numpy as np


def test_total_energy():
    te.initialize()
    nnr = te.get_nnr()
    nspin = te.get_nspin()
    # ngm = te.get_ngm()
    nat = te.get_nat()
    te.print_energies()

    positions = te.get_positions(nat)
    positions = positions + 0.01*np.random.random([3, nat])
    te.set_positions(positions, nat)
    te.print_energies()

    rho_of_r = te.get_rho_of_r(nnr, nspin)
    total_rho = np.sum(rho_of_r)
    rho_of_r = rho_of_r + 1.0e-3
    rho_of_r = rho_of_r * total_rho / np.sum(rho_of_r)
    te.set_rho_of_r(rho_of_r, nnr, nspin)
    te.print_energies()
    e_rho_times_v_hxc, e_hartree, e_xc, e_ewald = te.get_energies()
    print(e_rho_times_v_hxc, e_hartree, e_xc, e_ewald)
    return True


if __name__ == "__main__":

    # Run the tests.
    test_total_energy()
