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

def message(txt, bar="="*79):
    print(f"\n{bar}\n{txt}\n{bar}\n")


def train_network_ensemble(n_models=5):
    for idx in range(n_models):
        message(f"Train model {idx+1} / {n_models}")
        parameters = mala.Parameters()
        parameters.data.data_splitting_type = "by_snapshot"
        parameters.data.input_rescaling_type = "feature-wise-standard"
        parameters.data.output_rescaling_type = "normal"
        parameters.network.layer_activations = ["ReLU"]
        parameters.running.max_number_epochs = 10
        parameters.running.mini_batch_size = 40
        parameters.running.learning_rate = 0.00001
        parameters.running.trainingtype = "Adam"

        data_handler = mala.DataHandler(parameters)
        data_handler.add_snapshot("Be_snapshot0.in.npy", data_path,
                                  "Be_snapshot0.out.npy", data_path, "tr")
        data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                                  "Be_snapshot1.out.npy", data_path, "va")
        data_handler.prepare_data()
        printout("Read data: DONE.")

        parameters.network.layer_sizes = [
              data_handler.input_dimension,
              100,
              data_handler.output_dimension]

        network = mala.Network(parameters)
        trainer = mala.Trainer(parameters, network, data_handler)
        trainer.train_network()
        additional_calculation_data = os.path.join(data_path,
                                                   "Be_snapshot0.out")
        trainer.save_run(f"be_model_test{idx}",
                         additional_calculation_data=additional_calculation_data)


# Load a model.
if not os.path.exists("be_model_test0.zip"):
    train_network_ensemble()

# Set up ASE objects.
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

message("Calculate total energy for ensemble")
atoms.set_calculator(calculator)
atoms.get_potential_energy()
print(calculator.results["energy"],
      calculator.results["energy_std"],
      calculator.results["energy_samples"])
