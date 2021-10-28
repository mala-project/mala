"""ASE calculator for MALA predictions."""

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from mpi4py import MPI
from mala import Parameters, Network, DataHandler, Predictor, LDOS, Density, \
                 DOS
from mala.common.parallelizer import get_rank, get_comm
import total_energy as te

class ASECalculator(Calculator):
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
        [0]: A type of additional calculation data
        [1]: A path to it.
        With this additonal calculation data (preferably from the training of
        the neural network), calculator can access all important data such as
        temperature, number of electrons, etc. that might not be known simply
        from the atomic positions.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, params: Parameters, network: Network,
                 data: DataHandler, reference_data):
        super(ASECalculator, self).__init__()

        # Copy the MALA relevant objects.
        self.params: Parameters = params
        if self.params.targets.target_type != "LDOS":
            raise Exception("The MALA calculator currently only works with the"
                            "LDOS.")

        self.network: Network = network
        self.data_handler: DataHandler = data

        # Prepare for prediction.
        self.predictor = Predictor(self.params, self.network,
                                   self.data_handler)

        # Get critical values from a reference file (cutoff, temperature, etc.)
        self.data_handler.target_calculator.\
            read_additional_calculation_data(reference_data[0],
                                             reference_data[1])

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
        get_comm().Barrier()
        # Calculator.calculate(self, atoms, properties, system_changes)

        # Get the LDOS from the NN.
        ldos = self.predictor.predict_for_atoms(atoms)

        energy = 0.0
        forces = np.zeros([len(atoms), 3], dtype=np.float64)
        if self.params.use_mpi:
            # TODO: For whatever reason the ase.io.read command we issue
            #  during the
            # first initilization of the TEM causes an error when used in
            # parallel. I have tracked the issue to this command and I think
            # it is stemming from ASE itself. Apparently, ASE detects
            # present MPI environmnents, and tries to use them. I am guessing
            # this goes wrong, because we only call the TEM on rank 0.
            # We could mitigate this by calling the file I/O on all ranks
            # (possible, but requires some changes in the interface) or,
            # for simplicity simply provide the file ourselves.
            create_file = False
        else:
            create_file = True
        if get_rank() == 0:
            # Define calculator objects.
            ldos_calculator: LDOS = self.data_handler.target_calculator
            density_calculator = Density.from_ldos(ldos_calculator)
            dos_calculator = DOS.from_ldos(ldos_calculator)

            # Get DOS and density.
            dos = ldos_calculator.get_density_of_states(ldos)
            fermi_energy_ev = dos_calculator.get_self_consistent_fermi_energy_ev(
                dos)
            density = ldos_calculator.get_density(ldos,
                                                  fermi_energy_ev=fermi_energy_ev)
            energy = ldos_calculator.\
            get_total_energy(dos_data=dos, density_data=density,
                             fermi_energy_eV=fermi_energy_ev,
                             create_qe_file=create_file)
            forces = density_calculator.get_atomic_forces(density,
                                                          create_file=create_file)
        print("Before Final barrier")
        get_comm().Barrier()
        print("After Final barrier")
        energy = get_comm().bcast(energy, root=0)
        get_comm().Bcast([forces, MPI.DOUBLE], root=0)
        print(get_rank(), forces)
        # Use the LDOS determined DOS and density to get energy and forces.
        self.results["energy"] = energy
        self.results["forces"] = forces
