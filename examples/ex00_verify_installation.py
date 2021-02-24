import fesl
from fesl import printout
import numpy as np

"""
ex00_verify_installation.py: This example confirms whether or not your setup is correct.  
"""

fesl.printout("Welcome to FESL.")
printout("Running ex00_verify_installation.py")

test_parameters = fesl.Parameters()
test_descriptors = fesl.DescriptorInterface(test_parameters)
test_targets = fesl.TargetInterface(test_parameters)
test_handler = fesl.DataHandler(test_parameters, descriptor_calculator=test_descriptors, target_calculator=test_targets)
test_network = fesl.Network(test_parameters)
test_hpoptimizer = fesl.HyperOptInterface(test_parameters)
data_path = None
try:
    from data_repo_path import get_data_repo_path
    data_path = get_data_repo_path()
    test_array = np.load(data_path+"linking_tester.npy")
    printout("Current data repository path: ", data_path)
except FileNotFoundError:
    printout("It seems that you don't have a data repository set up properly. You can still use FESL, but tests and examples WILL fail.")
printout("Successfully ran ex00_verify_installation.py.")
printout("Congratulations, your installation seems to work!")
