import os

import mala
from mala import printout
from ase.io import read

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex19_ase_calculator_with_uncertainty.py: Shows how MALA can be used as an ASE 
calculator. Currently, calculation of forces is not supported. 

REQUIRES LAMMPS AND QUANTUM ESPRESSO (TOTAL ENERGY MODULE).
"""


def train_network_ensemble():
    for i in range(0, 5):
        test_parameters = mala.Parameters()
        test_parameters.data.data_splitting_type = "by_snapshot"
        test_parameters.data.input_rescaling_type = "feature-wise-standard"
        test_parameters.data.output_rescaling_type = "normal"
        test_parameters.network.layer_activations = ["ReLU"]
        test_parameters.running.max_number_epochs = 100
        test_parameters.running.mini_batch_size = 40
        test_parameters.running.learning_rate = 0.00001
        test_parameters.running.trainingtype = "Adam"

        data_handler = mala.DataHandler(test_parameters)
        data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                  "Be_snapshot0.out.npy", data_path, "tr")
        data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                  "Be_snapshot1.out.npy", data_path, "va")
        data_handler.prepare_data()
        printout("Read data: DONE.")

        test_parameters.network.layer_sizes = [
              data_handler.input_dimension,
              100,
              data_handler.output_dimension]

        test_network = mala.Network(test_parameters)
        test_trainer = mala.Trainer(test_parameters, test_network,
                                    data_handler)
        test_trainer.train_network()
        additional_calculation_data = os.path.join(data_path,
                                                   "Be_snapshot0.out")
        test_trainer.save_run("be_model_test"+str(i),
                              additional_calculation_data=additional_calculation_data)


# train_network_ensemble()
# Load a model.
assert os.path.exists("be_model_test0.zip")

# Set up the ASE objects.
atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
calculator = mala.MALASimpleEnsemble.load_model("be_model_test*")

####################
# PARAMETERS
# Make sure all parameters are prepared for the run.
####################
calculator.mala_parameters.targets.target_type = "LDOS"
calculator.mala_parameters.targets.ldos_gridsize = 11
calculator.mala_parameters.targets.ldos_gridspacing_ev = 2.5
calculator.mala_parameters.targets.ldos_gridoffset_ev = -5

calculator.mala_parameters.descriptors.descriptor_type = "Bispectrum"
calculator.mala_parameters.descriptors.bispectrum_twojmax = 10
calculator.mala_parameters.descriptors.bispectrum_cutoff = 4.67637
calculator.mala_parameters.targets.pseudopotential_path = data_path
calculator.mala_parameters.targets.pseudopotential_path = data_path

atoms.set_calculator(calculator)
atoms.get_potential_energy()
print(calculator.results["energy"],
      calculator.results["energy_std"])

