from fesl_integration import check_analytical_integration, qe_dens_to_nr_of_electrons, qe_ldos_to_density, qe_ldos_to_dos
from tensor_memory import test_tensor_memory
from lazy_loading_basic import test_lazy_loading_basic
from lazy_loading_horovod_benchmark import lazy_loading_horovod_benchmark
from fesl.common.parameters import printout
from data_repo_path import get_data_repo_path
from inference_test import run_inference_test
data_path = get_data_repo_path()+"Al256_reduced/"


''''
fesl_tests: Central hub for all subroutine tests for fesl.
'''
standard_accuracy = 10**-10

######################
# Memory tests (obsolete).
######################

if test_tensor_memory(data_path+"Al_debug_2k_nr0.in.npy", standard_accuracy):
    printout("test_tensor_memory suceeded.")
else:
    raise Exception("test_tensor_memory failed.")

######################
# Integration tests.
######################

if check_analytical_integration(standard_accuracy):
    printout("check_analytical_integration suceeded.")
else:
    raise Exception("check_analytical_integration failed.")

# These tests operate on a coarser accuracy

if qe_dens_to_nr_of_electrons(standard_accuracy*(10**3)):
    printout("qe_dens_to_nr_of_electrons suceeded.")
else:
    raise Exception("qe_dens_to_nr_of_electrons failed.")
if qe_ldos_to_density(standard_accuracy*(10**7)):
    printout("qe_ldos_to_density suceeded.")
else:
    raise Exception("qe_ldos_to_density failed.")
if qe_ldos_to_dos(standard_accuracy*(10**4)):
    printout("qe_ldos_to_dos suceeded.")
else:
    raise Exception("qe_ldos_to_dos failed.")

######################
# Total energy test.
######################

# Find a way to access the total energy module form a runner.
# from total_energy_tester import test_total_energy
# if test_total_energy():
#     printout("test_total_energy suceeded.")
# else:
#     raise Exception("test_total_energy failed.")

######################
# Scaling test.
######################

if test_lazy_loading_basic(data_path=data_path):
    printout("test_lazy_loading suceeded.")
else:
    raise Exception("test_lazy_loading failed.")

######################
# Lazy loading and horovod tests. This cannot yet be included in the
# automated test suite because we don't install horovod there.
######################

try:
    if lazy_loading_horovod_benchmark(data_path=data_path):
        printout("test_lazy_loading suceeded.")
except:
    printout("Could not perform lazy loading horovod test, most likely because horovod was not installed.")

######################
# Inference tests.
######################

if run_inference_test():
    printout("run_inference_test test suceeded.")
else:
    raise Exception("run_inference_test failed.")