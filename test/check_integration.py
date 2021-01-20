import matplotlib.pyplot as plt
import numpy as np
from fesl.targets.ldos import LDOS
from fesl.targets.density import Density
from fesl.targets.dos import DOS
from fesl.common.parameters import Parameters

# In order to test the integration capabilities of FESL we need a QuantumEspresso
# calculation containing the following:
#   1. Outfile from the run
#   2. LDOS cube files.
#   3. Density cube file.
#   4. A DOS file.
# In the check_integration_qe subfolder, input scripts can be found  for a
# simple test calculation (the results of this calculation have been used
# to develop this script). It uses an Al supercell with 36 atoms and a
# 6x6x6 k-grid.

data_path = "/home/fiedlerl/data/Al36/for_fesl/"
path_to_out = data_path+"Al.pw.scf.out"
path_to_ldos_qe = [data_path, "tmp.pp*Al_ldos.cube"]
path_to_dos_qe = [data_path, "Al.dos"]
path_to_dens_qe = [data_path, "Al_dens.cube"]
path_to_ldos_npy = data_path+"Al_ldos.npy"
path_to_dos_npy = data_path+"Al_dos.npy"
path_to_dens_npy = data_path+"Al_dens.npy"

# We can read from numpy arrays or directly from QE data.
# In the later case, numpy arrays will be saved for the subsqeuent run.
numpy_arrays = True


# Define the error margins and the parameters used for the experiments.
rel_error_margin_number_of_electrons = 0.005
rel_error_margin_density = 0.005
rel_error_margin_dos = 0.005
test_parameters = Parameters()
test_parameters.targets.ldos_gridsize = 250
test_parameters.targets.ldos_gridspacing_ev = 0.1
test_parameters.targets.ldos_gridoffset_ev = -10


# Integrate the QE density (Al.dens) over spatial grid. Does this yield the correct number of electrons?
def qe_dens_to_nr_of_electrons():
    # Create a calculator.
    dens_calculator = Density(test_parameters)
    dens_calculator.read_additional_calculation_data("qe.out",path_to_out)

    # Read the input data.
    if numpy_arrays:
        density_dft = np.load(path_to_dens_npy)
    else:
        density_dft = dens_calculator.read_from_cube(path_to_dens_qe[1], path_to_dens_qe[0])
        np.save(path_to_dens_npy, density_dft)

    # Calculate the quantities we want to compare.
    nr_fesl = dens_calculator.get_number_of_electrons(density_dft)
    nr_dft = dens_calculator.number_of_electrons

    # Calculate relative error.
    rel_error = np.abs(nr_fesl-nr_dft) / nr_dft
    print("Relative error number of electrons: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < rel_error_margin_number_of_electrons:
        return True
    else:
        return False

# Integrate QE LDOS over energy grid. Does this yield the correct density (when compared to Al.dens)?
def qe_ldos_to_density():
    # Create a calculator.abs()
    ldos_calculator = LDOS(test_parameters)
    ldos_calculator.read_additional_calculation_data("qe.out",path_to_out)

    # Read the input data.
    density_dft = np.load(path_to_dens_npy)
    if numpy_arrays:
        ldos_dft = np.load(path_to_ldos_npy)
    else:
        ldos_dft = ldos_calculator.read_from_cube(path_to_ldos_qe[1], path_to_ldos_qe[0])

        # LDOS is in 1/Ry. DOS is in 1/eV.
        ldos_dft = ldos_calculator.convert_units(ldos_dft, "1/Ry")
        np.save(path_to_ldos_npy, ldos_dft)

    # Calculate the quantities we want to compare.
    density_fesl = ldos_calculator.get_density(ldos_dft)
    density_fesl_sum = density_fesl.sum()
    density_dft_sum = density_dft.sum()

    # Calculate relative error.
    rel_error = np.abs(density_fesl_sum-density_dft_sum) / density_dft_sum
    print("Relative error for sum of density: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < rel_error_margin_number_of_electrons:
        return True
    else:
        return False


# Integrate the QE LDOS over spatial grid. Does this yield the corrected LDOS (when compared to Al.dos)?
def qe_ldos_to_dos():
    # Create the necessary calculators.
    ldos_calculator = LDOS(test_parameters)
    ldos_calculator.read_additional_calculation_data("qe.out",path_to_out)
    dos_calculator = DOS(test_parameters)
    dos_calculator.read_additional_calculation_data("qe.out",path_to_out)

    # Read the input data.
    ldos_dft = np.load(path_to_ldos_npy)
    if numpy_arrays:
        dos_dft = np.load(path_to_dos_npy)
    else:
        # DOS is in 1/eV so no conversion necessary.
        dos_dft = dos_calculator.read_from_qe_dos_txt(path_to_dos_qe[1], path_to_dos_qe[0])
        np.save(path_to_dos_npy, dos_dft)

    # Calculate the quantities we want to compare.
    dos_fesl = ldos_calculator.get_density_of_states(ldos_dft)
    dos_fesl_sum = dos_fesl.sum()
    dos_dft_sum = dos_dft.sum()
    rel_error = np.abs(dos_fesl_sum-dos_dft_sum) / dos_dft_sum
    print("Relative error for sum of DOS: ", rel_error)

    # Check against the constraints we put upon ourselves.
    if rel_error < rel_error_margin_dos:
        return True
    else:
        return False




# Run the tests.
test1 = qe_dens_to_nr_of_electrons()
print("Integrate QE density over spatial grid and get correct number of electrons compared to QE - success?:", test1)
test1 = qe_ldos_to_density()
print("Integrate QE LDOS over energy grid and get correct density compared to QE - succes?:", test1)
test1 = qe_ldos_to_dos()
print("Integrate QE LDOS over spatial grid and get correct DOS compared to QE - success:", test1)
