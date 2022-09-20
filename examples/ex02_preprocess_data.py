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
test_parameters.descriptors.descriptor_type = "SNAP"
test_parameters.descriptors.twojmax = 6
test_parameters.descriptors.rcutfac = 4.67637
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

# Take care to choose the "add_snapshot" function correct for
# the type of data you want to preprocess.
data_converter.add_snapshot_qeout_cube(os.path.join(data_path, "Be.pw.scf.out"),
                                       os.path.join(data_path, "cubes/tmp.pp*Be_ldos.cube"),
                                       output_units="1/(Ry*Bohr^3)")

# Convert all the snapshots and save them in the current directory.
data_converter.convert_snapshots("./", naming_scheme="Be_snapshot*")

# If parts of the data have already been processed, the DataConverter class can
# also be used to convert the rest.
# No matter which way you access the DataConvert, you can always specify
# keywords (check API) for the calculators.
data_converter = mala.DataConverter(test_parameters)
data_converter.add_snapshot_qeout(os.path.join(data_path, "Be.pw.scf.out"),)
data_converter.convert_snapshots("./", naming_scheme="Be_snapshot_only_in*",
                                 descriptor_calculation_kwargs={"working_directory": data_path})

data_converter = mala.DataConverter(test_parameters)
data_converter.add_snapshot_cube(os.path.join(data_path,
                                              "cubes/tmp.pp*Be_ldos.cube"),
                                 output_units="1/(Ry*Bohr^3)")
data_converter.convert_snapshots("./", naming_scheme="Be_snapshot_only_out*")

printout("Parameters used for this experiment:")
test_parameters.show()
