import os

from ase.io import read
import mala
from mala import printout

from mala.datahandling.data_repo import data_path

"""
Show how a prediction can be made using MALA, based on only a trained network and atomic
configurations. Either execute ex01 before executing this one or download the
appropriate model from the provided test data repo.

REQUIRES LAMMPS (and potentiall the total energy module).
"""

model_name = "Be_model"
model_path = "./" if os.path.exists("Be_model.zip") else data_path


####################
# 1. LOADING A NETWORK
# To use the predictor class to test an ML-DFT model, simply load it via the
# Tester class interface. Afterwards, set the necessary parameters.
####################
parameters, network, data_handler, predictor = mala.Predictor.load_run(
    run_name=model_name, path=model_path
)


####################
# 2. MAKING A PREDICTION
# Load atomic positions (e.g. from a calculation output file, an MD run, etc.).
# Predict the LDOS from it, and then use the LDOS calculator to calculate
# quantities of interest.
####################
atoms = read(os.path.join(data_path, "Be_snapshot3.out"))
ldos = predictor.predict_for_atoms(atoms)
ldos_calculator: mala.LDOS = predictor.target_calculator
ldos_calculator.read_from_array(ldos)
printout("Predicted band energy: ", ldos_calculator.band_energy)
# If the total energy module is installed, the total energy can also be
# calculated.
# parameters.targets.pseudopotential_path = data_path
# printout("Predicted total energy", ldos_calculator.total_energy)
