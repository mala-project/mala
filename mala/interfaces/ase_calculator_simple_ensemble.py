"""ASE calculator for MALA predictions."""

from ase.calculators.calculator import Calculator, all_changes
import numpy as np

from mala import Parameters, Network, DataHandler, LDOS, SimpleEnsemblePredictor
from mala.common.parallelizer import get_rank, get_comm, barrier
from mala.interfaces.ase_calculator import MALA


class MALASimpleEnsemble(MALA):
    """
    Implements an ASE calculator based on MALA predictions.

    With this, MD simulations can be performed.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Interface.

    networks : list
        List of networks to be used for the predictions.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler; not really used in the classical sense, more as a
        collector of properties and objects (such as a target calculator).

    reference_data : list
        A list containing

        - [0]: A type of additional calculation data
        - [1]: A path to it.

        With this additonal calculation data (preferably from the training of
        the neural network), calculator can access all important data such as
        temperature, number of electrons, etc. that might not be known simply
        from the atomic positions.
    """

    implemented_properties = ['energy']

    def __init__(self, params: Parameters, networks: list,
                 data: DataHandler, reference_data=None,
                 predictor=None):
        self.mala_parameters: Parameters = params
        if self.mala_parameters.targets.target_type != "LDOS":
            raise Exception("The MALA calculator currently only works with the"
                            "LDOS.")

        if predictor is None:
            predictor = SimpleEnsemblePredictor(params, networks, data)
        else:
            predictor = predictor

        super(MALASimpleEnsemble, self).__init__(params, networks[0],
                                                 data,
                                                 reference_data=reference_data,
                                                 predictor=predictor)

    @classmethod
    def load_model(cls, run_name, path="./"):
        """
        Load a model to use for the calculator.

        Only supports zipped models with .json parameters. No legacy
        models supported.

        Parameters
        ----------
        run_name : str
            Name under which the model is saved.

        path : str
            Path where the model is saved.
        """
        loaded_params, loaded_network, \
            new_datahandler, loaded_runner = SimpleEnsemblePredictor.\
            load_run(run_name, path=path)
        calculator = cls(loaded_params, loaded_network, new_datahandler,
                         predictor=loaded_runner)
        return calculator

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """
        Perform the calculations.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object for which to perform the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.

        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.
        """
        barrier()
        Calculator.calculate(self, atoms, properties, system_changes)

        # Get the LDOS from the NN.
        ldos = self.predictor.predict_for_atoms(atoms)
        energies = []

        for m in range(0, self.predictor.number_of_models):
            # forces = np.zeros([len(atoms), 3], dtype=np.float64)

            # If an MPI environment is detected, ASE will use it for writing.
            # Therefore we have to do this before forking.
            self.data_handler.\
                target_calculator.\
                write_tem_input_file(atoms,
                                     self.data_handler.
                                     target_calculator.qe_input_data,
                                     self.data_handler.
                                     target_calculator.qe_pseudopotentials,
                                     self.data_handler.
                                     target_calculator.grid_dimensions,
                                     self.data_handler.
                                     target_calculator.kpoints)

            ldos_calculator: LDOS = self.data_handler.target_calculator

            ldos_calculator.read_from_array(ldos[m])
            energy, self.last_energy_contributions \
                = ldos_calculator.get_total_energy(return_energy_contributions=
                                                   True)
            energies.append(energy)
            barrier()

        self.results["energy"] = np.mean(energies)
        self.results["energy_std"] = np.std(energies)
        self.results["energy_samples"] = energies
        # if "forces" in properties:
        #     self.results["forces"] = forces

    def save_calculator(self, filename):
        """
        Save parameters used for this calculator.

        This is useful for e.g. checkpointing.

        Parameters
        ----------
        filename : string
            Path to file in which to store the Calculator.

        """
        raise Exception("Saving currently not implemented for this class.")

