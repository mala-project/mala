import os

from ase.io import read
import mala
from mala import printout
import numpy as np
import torch

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
ex19_at.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("Be_model.zip"), "Be model missing, run ex01 first."


def run_prediction(backprop=False):
    """
    This just runs a regular MALA prediction for a two-atom Beryllium model.
    """
    parameters, network, data_handler, predictor = mala.Predictor.load_run(
        "Be_model"
    )

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

        numerical_forces.append(
            (derivative_plus - derivative_minus) / delta_numerical
        )

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

        numerical_forces.append(
            (derivative_plus - derivative_minus) / delta_numerical
        )

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
            _, derivative_plus = ldos_calculator.get_total_energy(
                return_energy_contributions=True
            )
            derivative_plus = derivative_plus["e_hartree"]

            ldos_minus = ldos.copy()
            ldos_minus[point, i] -= delta_numerical * 0.5
            ldos_calculator.read_from_array(ldos_minus)
            _, derivative_minus = ldos_calculator.get_total_energy(
                return_energy_contributions=True
            )
            derivative_minus = derivative_minus["e_hartree"]
            numerical_forces.append(
                (derivative_plus - derivative_minus) / delta_numerical
            )

        print(mala_forces[point, :] / np.array(numerical_forces))


def check_input_gradient_scaling():
    # Load a model, set paramters.
    predictor: mala.Predictor
    parameters: mala.Parameters
    parameters, network, data_handler, predictor = mala.Predictor.load_run(
        "Be_ACE_model_FULLSCALING"
    )
    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5
    parameters.targets.restrict_targets = None
    parameters.descriptors.descriptors_contain_xyz = False
    parameters.descriptors.descriptor_type = "ACE"

    # Compute descriptors, only do further computations for one point.
    atoms1 = read("/home/fiedlerl/data/mala_data_repo/Be2/Be_snapshot1.out")
    descriptors, ngrid = (
        predictor.data.descriptor_calculator.calculate_from_atoms(
            atoms1, [18, 18, 27]
        )
    )
    snap_descriptors = torch.from_numpy(np.array([descriptors[0, 0, 0]]))
    snap_descriptors_work = snap_descriptors.clone()
    snap_descriptors = predictor.data.input_data_scaler.transform(
        snap_descriptors
    )
    snap_descriptors.requires_grad = True

    # Forward pass through the network.
    ldos = network(snap_descriptors)
    d_d_d_B = []

    # Compute "gradient". This is ACTUALLY the Jacobian matrix, i.e., the one
    # we can compare with finite differences.
    for i in range(0, 11):
        if snap_descriptors.grad is not None:
            snap_descriptors.grad.zero_()

        ldos[0, i].backward(retain_graph=True)
        # d_d_d_B = snap_descriptors.grad.clone()
        d_d_d_B.append(
            predictor.data.input_data_scaler.inverse_transform_backpropagation(
                snap_descriptors.grad.clone()
            )
        )

    # Just for debugging purposes, not part of the actual test.
    # if False:
    #     # This would be the direct application of the autograd.
    #     # Autograd computes the vector Jacobian product, see
    #     # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
    #     # Here: compare same_as_sum with d_d_d_B_sum.
    #     ldos = network(snap_descriptors)
    #     snap_descriptors.grad.zero_()
    #     # print(ldos / ldos)
    #     ldos.backward(ldos / ldos, retain_graph=True)
    #     same_as_sum = (
    #         predictor.data.input_data_scaler.inverse_transform_backpropagation(
    #             snap_descriptors.grad.clone()
    #         )
    #     )
    #     d_d_d_B_sum = d_d_d_B[0]
    #     for i in range(1, 11):
    #         d_d_d_B_sum += d_d_d_B[i]

    with torch.no_grad():
        # In case we want to compare different levels of finite differences.
        for diff in [
            # 5.0e-1,
            # 5.0e-2,
            # 5.0e-3,
            5.0e-4,
            # 5.0e-5,
            # 5.0e-6,
        ]:

            # Compute finite differences. j theoretically goes up to
            # 36, but for a simple check this is completely enough.
            for j in range(0, 5):
                snap_descriptors_work[0, j] += diff
                descriptors_scaled1 = snap_descriptors_work.clone()
                ldos_1 = network(
                    predictor.data.input_data_scaler.transform(
                        descriptors_scaled1
                    )
                ).clone()

                snap_descriptors_work[0, j] -= 2.0 * diff
                descriptors_scaled2 = snap_descriptors_work.clone()
                ldos_2 = network(
                    predictor.data.input_data_scaler.transform(
                        descriptors_scaled2
                    )
                ).clone()

                # Comparison - we only compare the first component of the
                # LDOS for brevity, but this works for all components.
                snap_descriptors_work[0, j] += diff
                force = -1.0 * ((ldos_2[0, 0] - ldos_1[0, 0]) / (2 * diff))
                print(
                    diff,
                    force.double().numpy(),
                    d_d_d_B[0][0, j],
                    force.double().numpy() / d_d_d_B[0][0, j],
                )


def check_output_gradient_scaling():
    # Load a model, set paramters.
    predictor: mala.Predictor
    parameters: mala.Parameters
    parameters, network, data_handler, predictor = mala.Predictor.load_run(
        "Be_ACE_model_FULLSCALING"
    )
    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5
    parameters.targets.restrict_targets = None
    parameters.descriptors.descriptors_contain_xyz = False
    parameters.descriptors.descriptor_type = "ACE"

    # Compute descriptors, only do further computations for one point.
    atoms1 = read("/home/fiedlerl/data/mala_data_repo/Be2/Be_snapshot1.out")
    descriptors, ngrid = (
        predictor.data.descriptor_calculator.calculate_from_atoms(
            atoms1, [18, 18, 27]
        )
    )
    snap_descriptors = torch.from_numpy(np.array([descriptors[0, 0, 0]]))
    snap_descriptors_work = snap_descriptors.clone()
    snap_descriptors = predictor.data.input_data_scaler.transform(
        snap_descriptors
    )
    snap_descriptors.requires_grad = True

    # Forward pass through the network.
    # ldos = network(snap_descriptors)
    # d_d_d_B = []

    # Compute "gradient". This is ACTUALLY the Jacobian matrix, i.e., the one
    # we can compare with finite differences.
    # for i in range(0, 11):
    #     if snap_descriptors.grad is not None:
    #         snap_descriptors.grad.zero_()
    #
    #     ldos[0, i].backward(retain_graph=True)
    #     # d_d_d_B = snap_descriptors.grad.clone()
    #     d_d_d_B.append(
    #         predictor.data.input_data_scaler.inverse_transform_backpropagation(
    #             snap_descriptors.grad.clone()
    #         )
    #     )

    if True:
        # This would be the direct application of the autograd.
        # Autograd computes the vector Jacobian product, see
        # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        ldos = network(snap_descriptors)
        ldos_unscaled = predictor.data.output_data_scaler.inverse_transform(
            ldos, copy=True
        )
        if snap_descriptors.grad is not None:
            snap_descriptors.grad.zero_()
        ldos.backward(
            predictor.data.output_data_scaler.transform_backpropagation(
                2 * ldos_unscaled
            ),
            retain_graph=True,
        )
        d_d_d_B = (
            predictor.data.input_data_scaler.inverse_transform_backpropagation(
                snap_descriptors.grad.clone()
            )
        )

    with torch.no_grad():
        # In case we want to compare different levels of finite differences.
        for diff in [
            # 5.0e-1,
            # 5.0e-2,
            # 5.0e-3,
            5.0e-4,
            # 5.0e-5,
            # 5.0e-6,
        ]:

            # Compute finite differences. j theoretically goes up to
            # 36, but for a simple check this is completely enough.
            for j in range(0, 5):
                snap_descriptors_work[0, j] += diff
                descriptors_scaled1 = snap_descriptors_work.clone()
                ldos_1 = predictor.data.output_data_scaler.inverse_transform(
                    network(
                        predictor.data.input_data_scaler.transform(
                            descriptors_scaled1
                        )
                    ).clone()
                )
                energy_1 = 0
                for i in range(0, 11):
                    energy_1 += ldos_1[0, i] * ldos_1[0, i]

                snap_descriptors_work[0, j] -= 2.0 * diff
                descriptors_scaled2 = snap_descriptors_work.clone()
                ldos_2 = predictor.data.output_data_scaler.inverse_transform(
                    network(
                        predictor.data.input_data_scaler.transform(
                            descriptors_scaled2
                        )
                    ).clone()
                )
                energy_2 = 0
                for i in range(0, 11):
                    energy_2 += ldos_2[0, i] * ldos_2[0, i]

                # Comparison - we only compare the first component of the
                # LDOS for brevity, but this works for all components.
                snap_descriptors_work[0, j] += diff
                force = -1.0 * ((energy_2 - energy_1) / (2 * diff))
                print(
                    diff,
                    force.double().numpy(),
                    d_d_d_B[0, j],
                    force.double().numpy() / d_d_d_B[0, j],
                )


# check_input_gradient_scaling()
# check_output_gradient_scaling()
# band_energy_contribution()
# entropy_contribution()
# hartree_contribution()
