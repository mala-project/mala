import os

import mala
from ase.io import read

from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."

"""
Shows how MALA can be used as an ASE calculator. 
Currently, calculation of forces is not supported. 

REQUIRES LAMMPS AND QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""


####################
# 1. LOADING A NETWORK
# To use the ASE calculator, just load a MALA ML-DFT model.
# Further make sure to set the path to the pseudopotential used during
# data generation-
####################
calculator = mala.MALA.load_model("be_model")
calculator.mala_parameters.targets.pseudopotential_path = data_path

####################
# 2. MAKING A PREDICTION
# A prediction via the ASE calculator can now be made by simply defining
# an ASE atoms object, setting the calculator and using the regular ASE
# interface.
####################
atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
atoms.set_calculator(calculator)
print(atoms.get_potential_energy())
