import os

from ase.io import read
import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex12_run_predictions.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."


def run_prediction(backprop=False):
    parameters, network, data_handler, predictor = mala.Predictor. \
        load_run("be_model")

    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5

    parameters.descriptors.descriptor_type = "Bispectrum"
    parameters.descriptors.bispectrum_twojmax = 10
    parameters.descriptors.bispectrum_cutoff = 4.67637
    parameters.targets.pseudopotential_path = data_path

    atoms = read(os.path.join(data_path, "Be_snapshot3.out"))
    ldos = predictor.predict_for_atoms(atoms, save_grads=backprop)
    ldos_calculator: mala.LDOS = predictor.target_calculator
    ldos_calculator.read_from_array(ldos)
    return ldos, ldos_calculator, parameters, predictor


def band_energy_contribution():
    ldos, ldos_calculator, parameters, predictor = run_prediction()
    ldos_calculator.debug_forces_flag = "band_energy"
    mala_forces = ldos_calculator.atomic_forces.copy()

    delta_numerical = 1e-6
    numerical_forces = []

    for i in range(0, parameters.targets.ldos_gridsize):
        ldos_plus = ldos.copy()
        ldos_plus[0, i] += delta_numerical * 0.5
        ldos_calculator.read_from_array(ldos_plus)
        derivative_plus = ldos_calculator.band_energy

        ldos_minus = ldos.copy()
        ldos_minus[0, i] -= delta_numerical * 0.5
        ldos_calculator.read_from_array(ldos_minus)
        derivative_minus = ldos_calculator.band_energy

        numerical_forces.append((derivative_plus - derivative_minus) /
                                delta_numerical)

    print("TEST BAND ENERGY FORCE CONTRIBUTIONS")
    print(mala_forces[0, :] / np.array(numerical_forces))
    print(mala_forces[2000, :] / np.array(numerical_forces))
    print(mala_forces[4000, :] / np.array(numerical_forces))


def entropy_contribution():
    ldos, ldos_calculator, parameters, predictor = run_prediction()
    ldos_calculator.debug_forces_flag = "entropy_contribution"
    mala_forces = ldos_calculator.atomic_forces.copy()

    delta_numerical = 1e-8
    numerical_forces = []

    for i in range(0, parameters.targets.ldos_gridsize):
        ldos_plus = ldos.copy()
        ldos_plus[0, i] += delta_numerical * 0.5
        ldos_calculator.read_from_array(ldos_plus)
        derivative_plus = ldos_calculator.entropy_contribution

        ldos_minus = ldos.copy()
        ldos_minus[0, i] -= delta_numerical * 0.5
        ldos_calculator.read_from_array(ldos_minus)
        derivative_minus = ldos_calculator.entropy_contribution

        numerical_forces.append((derivative_plus - derivative_minus) /
                                delta_numerical)

    print("TEST ENTROPY FORCE CONTRIBUTIONS")
    print(mala_forces[0, :] / np.array(numerical_forces))
    print(mala_forces[2000, :] / np.array(numerical_forces))
    print(mala_forces[4000, :] / np.array(numerical_forces))


def hartree_contribution_density_derivative():
    # This only tests dEh/dn
    ldos, ldos_calculator, parameters, predictor = run_prediction()
    density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
    density = ldos_calculator.density
    mala_forces = density_calculator.force_contributions
    ldos_calculator.debug_forces_flag = "hartree"

    delta_numerical = 1e-6
    points = [0, 1000, 2000, 3000, 4000, 5000]

    # numerical_forces = []
    for point in points:

        dens_plus = density.copy()
        dens_plus[point] += delta_numerical * 0.5
        density_calculator.read_from_array(dens_plus)
        derivative_plus = density_calculator.total_energy_contributions[
            "e_hartree"]

        dens_plus = density.copy()
        dens_plus[point] -= delta_numerical * 0.5
        density_calculator.read_from_array(dens_plus)
        derivative_minus = density_calculator.total_energy_contributions[
            "e_hartree"]

        # numerical_forces.append((derivative_plus - derivative_minus) /
        #                         delta_numerical)
        numerical_forces = (derivative_plus - derivative_minus) / \
                                delta_numerical

        print(mala_forces[point, :] / np.array(numerical_forces))


def hartree_contribution():
    ldos, ldos_calculator, parameters, predictor = run_prediction()
    density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
    density = ldos_calculator.density
    # mala_forces = density_calculator.force_contributions
    ldos_calculator.debug_forces_flag = "hartree"
    mala_forces = ldos_calculator.atomic_forces.copy()

    delta_numerical = 1e-6
    points = [0, 2000, 4000]

    for point in points:
        numerical_forces = []
        for i in range(0, parameters.targets.ldos_gridsize):
            ldos_plus = ldos.copy()
            ldos_plus[0, i] += delta_numerical * 0.5
            ldos_calculator.read_from_array(ldos_plus)
            _, derivative_plus = ldos_calculator.get_total_energy(return_energy_contributions=True)
            derivative_plus = derivative_plus["e_hartree"]

            ldos_minus = ldos.copy()
            ldos_minus[0, i] -= delta_numerical * 0.5
            ldos_calculator.read_from_array(ldos_minus)
            _, derivative_minus = ldos_calculator.get_total_energy(return_energy_contributions=True)
            derivative_minus = derivative_minus["e_hartree"]

            numerical_forces.append((derivative_plus - derivative_minus) /
                                    delta_numerical)

        # numerical_forces.append((derivative_plus - derivative_minus) /
        #                         delta_numerical)

        print(mala_forces[point, :] / np.array(numerical_forces))


def backpropagation():
    ldos, ldos_calculator, parameters, predictor = run_prediction(backprop=True)
    ldos_calculator.input_data_derivative = predictor.input_data
    ldos_calculator.output_data_torch = predictor.output_data
    mala_forces = ldos_calculator.atomic_forces
    print(mala_forces.size()) # Should be 8748, 91


# band_energy_contribution()
# entropy_contribution()
# hartree_contribution_density_derivative()
hartree_contribution()
# backpropagation()
