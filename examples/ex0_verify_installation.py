from mldft.common.parameters import Parameters
from mldft.datahandling.handler_interface import HandlerInterface
from mldft.targets.target_interface import TargetInterface
from mldft.network.network import Network

haslammpsinstalled = True
try:
    from mldft.descriptors.descriptor_interface import DescriptorInterface
except ModuleNotFoundError:
    haslammpsinstalled = False


"""
ex0_verify_installation.py: This example confirms whether or not your setup is correct.  
"""

print("Welcome to ML-DFT@CASUS.")
print("Running ex0_verify_installation.py")

test_parameters = Parameters()
test_targets = TargetInterface(test_parameters)
if haslammpsinstalled:
    test_descriptors = DescriptorInterface(test_parameters)
    test_handler = HandlerInterface(test_parameters, test_descriptors, test_targets)
else:
    test_handler = HandlerInterface(test_parameters)
test_network = Network(test_parameters)

print("Successfully ran ex0_verify_installation.py.")
if haslammpsinstalled is False:
    print("It looks like you don't have LAMMPS installed on your machine. You can still use most of the "
          "functionalities of this module without problems, but are limited to preprocessed data. If you try to use a "
          "descriptor class, it will most likely fail.")
print("Congratulations, your installation seems to work!")
