import os

from ase.io import read
import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex19_at.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."


def run_prediction(backprop=False):
    """
    This just runs a regular MALA prediction for a two-atom Beryllium model.
    """
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
    """
    Test the band energy contribution to the forces by calculating it
    via finite differences and analytically. This should return arrays of 1s.
    Note that the LDOS is only shifted at one point in space, but since the
    band energy is a not a spatially resolved quantity, the forces should
    be the same for all grid points.
    """
    ldos, ldos_calculator, parameters, predictor = run_prediction()

    # Only compute a specific part of the forces.
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

    print("FORCE TEST: Band energy contribution.")
    print(mala_forces[0, :] / np.array(numerical_forces))
    print(mala_forces[2000, :] / np.array(numerical_forces))
    print(mala_forces[4000, :] / np.array(numerical_forces))


def entropy_contribution():
    """
    Test the entropy contribution to the forces by calculating it
    via finite differences and analytically. This should return arrays of 1s,
    it will most likely not for many test settings due to the fact that the
    entropy contribution is VERY small.
    Note that the LDOS is only shifted at one point in space, but since the
    entropy is a not a spatially resolved quantity, the forces should
    be the same for all grid points.
    """
    ldos, ldos_calculator, parameters, predictor = run_prediction()

    # Only compute a specific part of the forces.
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

    print("FORCE TEST: Entropy contribution.")
    print(mala_forces[0, :] / np.array(numerical_forces))
    print(mala_forces[2000, :] / np.array(numerical_forces))
    print(mala_forces[4000, :] / np.array(numerical_forces))


def hartree_contribution():
    """
    Test the Hartree contribution to the forces by calculating it
    via finite differences and analytically. This should return arrays of 1s.
    Since the Hartree energy is dependent on the electronic density, the
    derivative will vary across space, and we have to compute the forces
    at multiple points.
    """
    ldos, ldos_calculator, parameters, predictor = run_prediction()

    # Only compute a specific part of the forces.
    ldos_calculator.debug_forces_flag = "hartree"
    mala_forces = ldos_calculator.atomic_forces.copy()

    delta_numerical = 1e-6
    points = [0, 2000, 4000]

    print("FORCE TEST: Hartree contribution.")
    for point in points:
        numerical_forces = []
        for i in range(0, parameters.targets.ldos_gridsize):
            ldos_plus = ldos.copy()
            ldos_plus[point, i] += delta_numerical * 0.5
            ldos_calculator.read_from_array(ldos_plus)
            _, derivative_plus = ldos_calculator.get_total_energy(return_energy_contributions=True)
            derivative_plus = derivative_plus["e_hartree"]

            ldos_minus = ldos.copy()
            ldos_minus[point, i] -= delta_numerical * 0.5
            ldos_calculator.read_from_array(ldos_minus)
            _, derivative_minus = ldos_calculator.get_total_energy(return_energy_contributions=True)
            derivative_minus = derivative_minus["e_hartree"]
            numerical_forces.append((derivative_plus - derivative_minus) /
                                    delta_numerical)

        print(mala_forces[point, :] / np.array(numerical_forces))


def backpropagation():
    """
    Test whether backpropagation works. To this end, the entire forces are
    computed, and then backpropagated through the network.
    """
    # Only compute a specific part of the forces.
    ldos, ldos_calculator, parameters, predictor = run_prediction(backprop=True)
    ldos_calculator.input_data_derivative = predictor.input_data
    ldos_calculator.output_data_torch = predictor.output_data
    mala_forces = ldos_calculator.atomic_forces
    # Should be 8748, 91
    print("FORCE TEST: Backpropagation machinery.")
    print(mala_forces.size())


band_energy_contribution()
entropy_contribution()
hartree_contribution()
backpropagation()
