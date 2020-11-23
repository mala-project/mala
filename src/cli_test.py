from common.parameters import parameters


'''
This framework should be easily usable by instantiating and calling a couple of classes.
This file is the central testing point to try out all new functions, as it serves as a command-line based testing interface.
'''

print("THIS IS ALL STILL WORK IN PROGRESS.")

####################
# PARSING
# TODO: Add parser.
####################



####################
# PARAMETERS
# All paramaters are handled from a central parameters class that contains subclasses.
####################
used_parameters = parameters()
# Modify parameters as you want...
# used_parameters.network.nn_type = "A different neural network"



####################
# DATA
# Read data to numpy arrays using OpenPMD.
# For the fingerprints, snapshot creation has to take place here.
####################



####################
# NETWORK SETUP
# Set up the network we want to use.
####################

####################
# TRAINING
# Set up the network we want to use.
####################

####################
# SAVING
# Save the network.
####################

####################
# INFERENCE
# Load a network and atomic configuration.
# Create the snapshot data and give it to the network,
# obtaining the LDOS.
####################

####################
# (optional) POSTPROCESSING
# Unclear if this can be done with this framework.
####################
