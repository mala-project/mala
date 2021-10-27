"""ASE calculator for MALA predictions."""

from ase.calculators.calculator import Calculator, all_changes

from mala import Parameters, Network, DataHandler, Predictor, LDOS, Density, \
                 DOS


class ASECalculator(Calculator):
    """
    Implements an ASE calculator based on MALA predictions.

    With this, MD simulations can be performed.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this Tester object.

    network : mala.network.network.Network
        Network which is being tested.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the test data.

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
        Calculator.calculate(self, atoms, properties, system_changes)

        # Get the LDOS from the NN.
        ldos = self.predictor.predict_for_atoms(atoms)

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

        # Use the LDOS determined DOS and density to get energy and forces.
        self.results["energy"] = ldos_calculator.\
            get_total_energy(dos_data=dos, density_data=density,
                             fermi_energy_eV=fermi_energy_ev)
        self.results["forces"] = density_calculator.get_atomic_forces(density)
