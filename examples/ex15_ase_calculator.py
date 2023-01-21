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


# Use an ASE calculator to make predictions.
def use_calculator(network, new_parameters, iscaler, oscaler):
    ####################
    # PARAMETERS
    # Make sure all parameters are prepared for the run.
    ####################
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5
    new_parameters.running.inference_data_grid = [18, 18, 27]

    new_parameters.descriptors.descriptor_type = "Bispectrum"
    new_parameters.descriptors.bispectrum_twojmax = 10
    new_parameters.descriptors.bispectrum_cutoff = 4.67637
    new_parameters.targets.pseudopotential_path = os.path.join(data_repo_path,
                                                               "Be2")

    # Construct a DataHandler object.
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Set up the ASE objects.
    atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
    calculator = mala.MALA(new_parameters, network,
                           inference_data_handler,
                           reference_data=
                                    ["espresso-out",
                                     os.path.join(data_path,
                                                  "Be_snapshot1.out")])
    atoms.set_calculator(calculator)
    print(atoms.get_potential_energy())


# Instead of training, these can be loaded.
parameters = mala.Parameters.load_from_file("be_model.params.json")
input_scaler = mala.DataScaler.load_from_file("be_model.iscaler.pkl")
output_scaler = mala.DataScaler.load_from_file("be_model.oscaler.pkl")
network = mala.Network.load_from_file(parameters, "be_model.network.pth")

# Next, use these values to run an MD simulation.
use_calculator(network, parameters, input_scaler, output_scaler)

