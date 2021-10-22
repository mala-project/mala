from ase.calculators.calculator import Calculator, all_changes

from mala import Parameters, Network, DataHandler, Predictor, LDOS, Density


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

    # TODO: Get rid of pp_valence_ectrons
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
        # Get the LDOS from the NN.
        ldos = self.predictor.predict_for_atoms(atoms)

        # Set additional parameters.
        ldos_calculator: LDOS = self.data_handler.target_calculator

        # use the LDOS to determine energies and/or forces.
        fermi_energy_ev = ldos_calculator.\
            get_self_consistent_fermi_energy_ev(ldos)
        if "energy" in properties:
            self.results["energy"] = ldos_calculator.\
                get_total_energy(ldos, fermi_energy_eV=fermi_energy_ev)
        if "forces" in properties:
            # For now, only the Hellman-Feynman forces can be calculated.
            density_calculator = Density.from_ldos(ldos_calculator)
            density = ldos_calculator.get_density(ldos, fermi_energy_ev=fermi_energy_ev)
            self.results["forces"] = density_calculator.get_atomic_forces(density)




