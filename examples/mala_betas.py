import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from ase.io import read, write
from ase import Atoms
import mala
from lammps import lammps
import torch
import numpy as np

torch.set_num_threads(1)
from mala.descriptors.lammps_utils import extract_compute_np
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")


def run_prediction(backprop=False, atoms=None, pass_descriptors=None):
    """
    This just runs a regular MALA prediction for a two-atom Beryllium model.
    """
    parameters, network, data_handler, predictor = mala.Predictor.load_run(
        "Be_ACE_model"
    )

    parameters.targets.target_type = "LDOS"
    parameters.targets.ldos_gridsize = 11
    parameters.targets.ldos_gridspacing_ev = 2.5
    parameters.targets.ldos_gridoffset_ev = -5

    parameters.descriptors.descriptor_type = "ACE"

    grd_loc = [18, 18, 27]
    if type(atoms) == str:
        atoms = read(atoms)
    else:
        atoms = atoms
    predictor.parameters.inference_data_grid = grd_loc
    assert atoms != None, "need to supply ase atoms object or file"
    if type(atoms) == str:
        atoms = read(atoms)
    else:
        atoms = atoms
    ldos = predictor.predict_for_atoms(
        atoms, save_grads=backprop, pass_descriptors=pass_descriptors
    )
    ldos_calculator: mala.LDOS = predictor.target_calculator
    ldos_calculator.read_from_array(ldos)
    ldos_calculator.read_additional_calculation_data(
        os.path.join(data_path, "Be_snapshot1.out")
    )
    return ldos, ldos_calculator, parameters, predictor


# boxlo, boxhi, xy, yz, xz, periodicity, box_change
def lammps_box_2_ASE_cell(lmpbox):
    Lx = lmpbox[1][0] - lmpbox[0][0]
    Ly = lmpbox[1][1] - lmpbox[0][1]
    Lz = lmpbox[1][2] - lmpbox[0][2]
    xy = lmpbox[2]
    yz = lmpbox[3]
    xz = lmpbox[4]
    a = [Lx, 0, 0]
    b = [xy, Ly, 0]
    c = [xz, yz, Lz]
    cel = [a, b, c]
    return cel


def lammps_2_ase_atoms(lmp, typ_map):
    cell = lammps_box_2_ASE_cell(lmp.extract_box())
    x = lmp.extract_atom("x")
    natoms = lmp.get_natoms()
    pos = np.array([[x[i][0], x[i][1], x[i][2]] for i in range(natoms)])
    # Extract atom types
    atom_types = lmp.extract_atom("type")
    # Convert atom types to NumPy array
    atom_types_lst = [atom_types[i] for i in range(natoms)]
    atom_syms = [typ_map[typi] for typi in atom_types_lst]
    atoms = Atoms(atom_syms)
    atoms.positions = pos
    atoms.set_cell(cell)
    atoms.set_pbc(True)  # assume pbc
    return atoms


def pre_force_callback(lmp):
    # gc.collect()
    L = lammps(ptr=lmp)
    """
    Test whether backpropagation works. To this end, the entire forces are
    computed, and then backpropagated through the network.
    """
    # Only compute a specific part of the forces.
    atoms = lammps_2_ase_atoms(L, typ_map={1: "Be"})
    write("test_mala_lammps.vasp", atoms)
    local_size = (18, 18, 27)
    nx, ny, nz = (18, 18, 27)
    feature_length = 36  # 5
    fingerprint_length = feature_length + 3
    ace_descriptors_np = extract_compute_np(
        L,
        "mygrid",
        0,
        2,
        (nz, ny, nx, fingerprint_length),
        use_fp64=False,
    )
    ace_descriptors_np = ace_descriptors_np.transpose([2, 1, 0, 3])
    print(fingerprint_length, np.shape(ace_descriptors_np))
    pass_descriptors = (
        ace_descriptors_np,
        local_size,
        fingerprint_length,
        True,
    )
    print("ace descs", ace_descriptors_np[1][1][1])
    print("positions", atoms.positions)
    ldos, ldos_calculator, parameters, predictor = run_prediction(
        backprop=True, atoms=atoms, pass_descriptors=pass_descriptors
    )
    ldos_calculator.debug_forces_flag = "band_energy"
    ldos_calculator.setup_for_forces(predictor)
    ldos_calculator.read_from_array(ldos)
    ldos_calculator.read_additional_calculation_data(
        os.path.join(data_path, "Be_snapshot1.out")
    )
    mala_forces = ldos_calculator.atomic_forces.copy()
    # energy attempt
    eng = ldos_calculator.band_energy
    # L.fix_external_set_energy_global('5', eng)
    # end energy attempt
    mala_forces = np.nan_to_num(mala_forces)
    mala_test = mala_forces.reshape(27, 18, 18, feature_length)
    # mala_test = mala_forces.reshape(18,18,27,feature_length)
    mala_test = mala_test.transpose([2, 1, 0, 3])
    print("mala_betas", mala_test[1][1][1])
    print(
        "mala force coeffs info:",
        mala_forces.shape,
        mala_test.shape,
        np.amax(mala_forces),
        np.mean(mala_forces),
    )
    print('mala "energy"', eng)
    mala_2_lammps = mala_test.flatten()
    L.close()
    return np.ascontiguousarray(mala_2_lammps)
