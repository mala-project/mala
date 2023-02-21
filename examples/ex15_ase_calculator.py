import os

import mala
from mala import printout
from ase.io import read

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex15_ase_calculator.py: Shows how MALA can be used as an ASE calculator. 
Currently, calculation of forces is not supported. 

REQUIRES LAMMPS AND QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""


# Load a model.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."
parameters, network, data_handler, predictor = mala.Predictor.\
    load_run("be_model")

####################
# PARAMETERS
# Make sure all parameters are prepared for the run.
####################
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5
parameters.running.inference_data_grid = [18, 18, 27]

parameters.descriptors.descriptor_type = "Bispectrum"
parameters.descriptors.bispectrum_twojmax = 10
parameters.descriptors.bispectrum_cutoff = 4.67637
parameters.targets.pseudopotential_path = os.path.join(data_repo_path, "Be2")

# Set up the ASE objects.
atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
calculator = mala.MALA(parameters, network,
                       data_handler,
                       reference_data=
                                ["espresso-out",
                                 os.path.join(data_path,
                                              "Be_snapshot1.out")])
atoms.set_calculator(calculator)
print(atoms.get_potential_energy())

