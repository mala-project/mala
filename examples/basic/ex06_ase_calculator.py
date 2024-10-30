import os

from ase.io import read
import mala

from mala.datahandling.data_repo import data_path

"""
Shows how MALA can be used as an ASE calculator.
Currently, calculation of forces is not supported. Either execute ex01 before executing
this one or download the appropriate model from the provided test data repo.

REQUIRES LAMMPS AND QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""

model_name = "Be_model"
model_path = "./" if os.path.exists("Be_model.zip") else data_path


####################
# 1. LOADING A NETWORK
# To use the ASE calculator, just load a MALA ML-DFT model.
# Further make sure to set the path to the pseudopotential used during
# data generation-
####################
calculator = mala.MALA.load_model(run_name=model_name, path=model_path)
calculator.mala_parameters.targets.pseudopotential_path = data_path

####################
# 2. MAKING A PREDICTION
# A prediction via the ASE calculator can now be made by simply defining
# an ASE atoms object, setting the calculator and using the regular ASE
# interface.
####################
atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
atoms.set_calculator(calculator)
mala.printout(atoms.get_potential_energy())
