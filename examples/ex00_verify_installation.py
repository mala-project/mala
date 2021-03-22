import mala
from mala import printout
import numpy as np

"""
ex00_verify_installation.py: This example confirms whether or not your setup 
is correct.  
"""

printout("Welcome to MALA.")
printout("Running ex00_verify_installation.py")

test_parameters = mala.Parameters()
test_descriptors = mala.DescriptorInterface(test_parameters)
test_targets = mala.TargetInterface(test_parameters)
test_handler = mala.DataHandler(test_parameters,
                                descriptor_calculator=test_descriptors,
                                target_calculator=test_targets)
test_network = mala.Network(test_parameters)
test_hpoptimizer = mala.HyperOptInterface(test_parameters, test_handler)
data_path = None
try:
    from data_repo_path import get_data_repo_path
    data_path = get_data_repo_path()
    test_array = np.load(data_path+"linking_tester.npy")
    printout("Current data repository path: ", data_path)
except FileNotFoundError:
    printout("It seems that you don't have a data repository set up properly. "
             "You can still use MALA, but tests and examples WILL fail.")
printout("Successfully ran ex00_verify_installation.py.")
printout("Congratulations, your installation seems to work!")
