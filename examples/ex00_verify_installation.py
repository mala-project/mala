from fesl.common.parameters import Parameters
from fesl.common.parameters import printout
from fesl.datahandling.data_handler import DataHandler
from fesl.network.network import Network
from fesl.network.hyper_opt_interface import HyperOptInterface
from fesl.descriptors.descriptor_interface import DescriptorInterface
from fesl.targets.target_interface import TargetInterface
import warnings
import numpy as np

"""
ex00_verify_installation.py: This example confirms whether or not your setup is correct.  
"""

printout("Welcome to FESL.")
printout("Running ex00_verify_installation.py")

test_parameters = Parameters()
test_descriptors = DescriptorInterface(test_parameters)
test_targets = TargetInterface(test_parameters)
test_handler = DataHandler(test_parameters, descriptor_calculator=test_descriptors, target_calculator=test_targets)
test_network = Network(test_parameters)
test_hpoptimizer = HyperOptInterface(test_parameters)
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
