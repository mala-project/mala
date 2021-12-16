import os

import mala
from mala import printout
from ase.io import read
from ase.md.nptberendsen import NVTBerendsen
from ase.md import MDLogger
from ase.units import fs
from mala.common.parallelizer import get_rank

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

"""
ex14_mala_md_parallel.py: Shows how MALA can be used to run a surrogate model 
based MD simulation, using a parallel implementation of the descriptor 
calculation in the LAMMPS code. Please run ex13 prior to this example, as 
the network from ex13 is being loaded here.
Please note that a non-local PP is being used, while the 
implementation of HF forces in MALA relies on local PPs and  
the network training is far from rigorous here. 
For production runs, make sure to a local PP and a properly trained and 
verified network to avoid problems during the MD run.

IMPORTANT: This example will currently only run with the version of the TEM 
that supports Hellmann-Feynman forces. If you are not sure if you have loaded
the correct version, contact the devs for further clarification.
"""


# Run a MALA MD.
def md_mala(network, new_parameters, iscaler, oscaler):
    ####################
    # PARAMETERS
    # Make sure all parameters are prepared for the run.
    ####################
    new_parameters.use_mpi = True
    new_parameters.targets.target_type = "LDOS"
    new_parameters.targets.ldos_gridsize = 11
    new_parameters.targets.ldos_gridspacing_ev = 2.5
    new_parameters.targets.ldos_gridoffset_ev = -5
    new_parameters.running.inference_data_grid = [18, 18, 27]

    new_parameters.descriptors.descriptor_type = "SNAP"
    new_parameters.descriptors.twojmax = 10
    new_parameters.descriptors.rcutfac = 4.67637
    new_parameters.targets.pseudopotential_path = os.path.join(data_repo_path,
                                                               "Be2")

    # Construct a DataHandler object.
    inference_data_handler = mala.DataHandler(new_parameters,
                                              input_data_scaler=iscaler,
                                              output_data_scaler=oscaler)

    # Set up the ASE objects.
    atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
    calculator = mala.ASECalculator(new_parameters, network,
                                    inference_data_handler,
                                    reference_data=
                                    ["qe.out",
                                     os.path.join(data_path,
                                                  "Be_snapshot1.out")])
    atoms.set_calculator(calculator)

    # Define and run the MD.
    dyn = NVTBerendsen(atoms, 1*fs, 298, 10)
    if get_rank() == 0:
        dyn.attach(MDLogger(dyn, atoms, 'mala_md.log', mode="w"), interval=1)
    printout("Starting the MD run")
    dyn.run(100)


if __name__ == "__main__":
    # Instead of training, these can also be loaded.
    parameters = mala.Parameters.load_from_file("ex13.params.pkl")
    input_scaler = mala.DataScaler.load_from_file("ex13.iscaler.pkl")
    output_scaler = mala.DataScaler.load_from_file("ex13.oscaler.pkl")
    network = mala.Network.load_from_file(parameters, "ex13.network.pth")

    # Next, use these values to run an MD simulation.
    md_mala(network, parameters, input_scaler, output_scaler)
