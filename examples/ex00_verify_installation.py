from mldft.common.parameters import Parameters
from mldft.datahandling.handler_interface import HandlerInterface
from mldft.network.network import Network
from mldft.descriptors.descriptor_interface import DescriptorInterface
from mldft.targets.target_interface import TargetInterface


"""
ex00_verify_installation.py: This example confirms whether or not your setup is correct.  
"""

print("Welcome to ML-DFT@CASUS.")
print("Running ex0_verify_installation.py")

test_parameters = Parameters()
test_descriptors = DescriptorInterface(test_parameters)
test_targets = TargetInterface(test_parameters)
test_handler = HandlerInterface(test_parameters, descriptor_calculator=test_descriptors, target_parser=test_targets)
test_network = Network(test_parameters)

print("Successfully ran ex0_verify_installation.py.")
print("Congratulations, your installation seems to work!")
