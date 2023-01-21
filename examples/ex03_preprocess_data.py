import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex02_preprocess_data.py: Shows how this framework can be used to preprocess
data. Preprocessing here means converting raw DFT calculation output into 
numpy arrays of the correct size. For the input data, this means descriptor
calculation.

Further preprocessing steps (scaling, unit conversion) is done later. 

REQUIRES LAMMPS.
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

test_parameters = mala.Parameters()

# Specify input data options, i.e. which descriptors are calculated
# with which parameters. These parameters are slightly modified for better
# performance.
test_parameters.descriptors.descriptor_type = "Bispectrum"
test_parameters.descriptors.bispectrum_twojmax = 6
test_parameters.descriptors.bispectrum_cutoff = 4.67637
test_parameters.descriptors.descriptors_contain_xyz = True

# Specify output data options, i.e. how the LDOS is parsed.
# The Be system used as an example here actually was calculated with
# drastically reduced number of energy levels for better computability.
test_parameters.targets.target_type = "LDOS"
test_parameters.targets.ldos_gridsize = 11
test_parameters.targets.ldos_gridspacing_ev = 2.5
test_parameters.targets.ldos_gridoffset_ev = -5

####################
# DATA
# Create a DataConverter, and add snapshots to it.
####################

data_converter = mala.DataConverter(test_parameters)

outfile = os.path.join(data_path, "Be.pw.scf.out")
ldosfile = os.path.join(data_path, "cubes/tmp.pp*Be_ldos.cube")
# The add_snapshot function can be called with a lot of options to reflect
# the data to be processed.
# The standard way is this: specifying descriptor, target and additional info.
data_converter.add_snapshot(descriptor_input_type="espresso-out",
                            descriptor_input_path=outfile,
                            target_input_type=".cube",
                            target_input_path=ldosfile,
                            additional_info_input_type="espresso-out",
                            additional_info_input_path=outfile,
                            target_units="1/(Ry*Bohr^3)")

# Likewise, there are multiple ways to save data. One way is to specify
# separate paths for descriptors, target and info data.
# A unified path can also be provided (see below).
# Further, naming_scheme impacts how data is saved. Standard is
# numpy, which can be indicated by ".npy".
# One can just as easily save using OpenPMD (if installed on your machine),
# in which case simply the correct file ending (e.g. ".h5" for HDF5) needs
# to be used on the naming_scheme.
data_converter.convert_snapshots(descriptor_save_path="./",
                                 target_save_path="./",
                                 additional_info_save_path="./",
                                 naming_scheme="Be_snapshot*.npy")

# If parts of the data have already been processed, the DataConverter class can
# also be used to convert the rest.
# No matter which way you access the DataConvert, you can always specify
# keywords (check API) for the calculators.
data_converter = mala.DataConverter(test_parameters)
data_converter.add_snapshot(descriptor_input_type="espresso-out",
                            descriptor_input_path=outfile,)
data_converter.convert_snapshots(complete_save_path="./",
                                 naming_scheme="Be_snapshot_only_in*",
                                 descriptor_calculation_kwargs=
                                 {"working_directory": data_path})

data_converter = mala.DataConverter(test_parameters)
data_converter.add_snapshot(target_input_type=".cube",
                            target_input_path=ldosfile,
                            target_units="1/(Ry*Bohr^3)")
data_converter.convert_snapshots(target_save_path="./",
                                 naming_scheme="Be_snapshot_only_out*")

printout("Parameters used for this experiment:")
test_parameters.show()
