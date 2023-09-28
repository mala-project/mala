"""ASE calculator for MALA predictions."""

from ase.calculators.calculator import Calculator, all_changes
import numpy as np

from mala import Parameters, Network, DataHandler, Predictor, LDOS, Density, \
                 DOS
from mala.common.parallelizer import get_rank, get_comm, barrier


class MALA(Calculator):
    """
    Implements an ASE calculator based on MALA predictions.

    With this, MD simulations can be performed.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Interface.

    network : mala.network.network.Network
        Network which is being used for the run.

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

    implemented_properties = ['energy', 'forces']

    def __init__(self, params: Parameters, network: Network,
                 data: DataHandler, reference_data=None,
                 predictor=None):
        super(MALA, self).__init__()

        # Copy the MALA relevant objects.
        self.mala_parameters: Parameters = params
        if self.mala_parameters.targets.target_type != "LDOS":
            raise Exception("The MALA calculator currently only works with the"
                            "LDOS.")

        self.network: Network = network
        self.data_handler: DataHandler = data

        # Prepare for prediction.
        if predictor is None:
            self.predictor = Predictor(self.mala_parameters, self.network,
                                       self.data_handler)
        else:
            self.predictor = predictor

        if reference_data is not None:
            # Get critical values from a reference file (cutoff,
            # temperature, etc.)
            self.data_handler.target_calculator.\
                read_additional_calculation_data(reference_data)

        # Needed for e.g. Monte Carlo.
        self.last_energy_contributions = {}

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
            new_datahandler, loaded_runner = Predictor.\
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

        ldos_calculator.read_from_array(ldos)
        energy, self.last_energy_contributions \
            = ldos_calculator.get_total_energy(return_energy_contributions=
                                               True)
        barrier()

        # Use the LDOS determined DOS and density to get energy and forces.
        self.results["energy"] = energy
        # if "forces" in properties:
        #     self.results["forces"] = forces

    def calculate_properties(self, atoms, properties):
        """
        After a calculation, calculate additional properties.

        This is separate from the calculate function because of
        MALA-MC simulations. For these energy and additional property
        calculation need to be separate.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object for which to perform the calculation.
            No needed per se, we can use it for a correctness check
            eventually.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of "rdf", ...
        """
        # TODO: Check atoms.

        if "rdf" in properties:
            self.results["rdf"] = self.data_handler.target_calculator.\
                get_radial_distribution_function(atoms)
        if "tpcf" in properties:
            self.results["tpcf"] = self.data_handler.target_calculator.\
                get_three_particle_correlation_function(atoms)
        if "static_structure_factor" in properties:
            self.results["static_structure_factor"] = self.data_handler.\
                target_calculator.get_static_structure_factor(atoms)
        if "ion_ion_energy" in properties:
            self.results["ion_ion_energy"] = self.\
                last_energy_contributions["e_ewald"]

    def save_calculator(self, filename, save_path="./"):
        """
        Save parameters used for this calculator.

        This is useful for e.g. checkpointing.

        Parameters
        ----------
        filename : string
            Name of the file in which to store the calculator.

        save_path : string
            Path where the calculator should be saved.

        """
        self.predictor.save_run(filename, save_path=save_path,
                                additional_calculation_data=True)

