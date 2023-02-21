import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex12_run_predictions.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."
parameters, network, data_handler, predictor = mala.Predictor.\
    load_run("be_model")

####################
# PARAMETERS
# Specify the correct LDOS parameters and load network.
####################
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5
parameters.running.inference_data_grid = [18, 18, 27]

parameters.descriptors.descriptor_type = "Bispectrum"
parameters.descriptors.bispectrum_twojmax = 10
parameters.descriptors.bispectrum_cutoff = 4.67637

# Using a QE out is simply for convenience; with the function
# .predict_from_atoms, prediction can be made directly from atomic
# configurations.
ldos = predictor.predict_from_qeout(os.path.join(data_path,
                                                 "Be_snapshot3.out"))
ldos_calculator: mala.LDOS = data_handler.target_calculator
ldos_calculator.read_from_array(ldos)
fermi_energy = ldos_calculator.get_self_consistent_fermi_energy(ldos)

printout("Predicted number of electrons: ", ldos_calculator.
         number_of_electrons)
printout("Predicted band energy: ", ldos_calculator.band_energy)
