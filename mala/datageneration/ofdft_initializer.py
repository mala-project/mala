from dftpy.api.api4ase import DFTpyCalculator
from dftpy.config import DefaultOption, OptionFormat
from ase import units
import ase.io
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


class OFDFTInitializer:
    """
    Initializes a trajectory using OF-DFT.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this instance.
    """

    def __init__(self, parameters, atoms):
        self.atoms = atoms
        self.params = parameters.datageneration

        # Check that only one element is used in the atoms.
        number_of_elements = len(set([x.symbol for x in self.atoms]))
        if number_of_elements > 1:
            raise Exception("OF-DFT-MD initialization can only work with one"
                            " element.")

    def get_equilibrated_configuration(self, logging_period=None):
        # Set the DFTPy configuration.
        conf = DefaultOption()
        conf['PATH']['pppath'] = self.params.local_psp_path
        conf['PP'][self.atoms[0].symbol] = self.params.local_psp_name
        conf['OPT']['method'] = 'TN'
        conf['KEDF']['kedf'] = 'WT'
        conf['JOB']['calctype'] = 'Energy Force'
        conf = OptionFormat(conf)
        # @Zhandos: Do we need this?
        # temperature = self.params.self.ofdft_temperature * units.kB

        # Create the DFTPy Calculator.
        calc = DFTpyCalculator(config=conf)
        self.atoms.set_calculator(calc)

        # Create the initial velocities, and dynamics object.
        MaxwellBoltzmannDistribution(self.atoms,
                                     temperature_K=
                                     self.params.ofdft_temperature,
                                     force_temp=True)
        dyn = Langevin(self.atoms, self.params.ofdft_timestep * units.fs,
                       temperature_K=self.params.ofdft_temperature,
                       friction=0.1)

        # If logging is desired, do the logging.
        if logging_period is not None:
            dyn.attach(MDLogger(dyn, self.atoms, 'mala_of_dft_md.log',
                                header=False, stress=False, peratom=True,
                                mode="a"), interval=logging_period)
            traj = Trajectory('mala_of_dft_md.traj', 'w', self.atoms)

            dyn.attach(traj.write, interval=logging_period)

        # Let the OF-DFT-MD run.
        ase.io.write("POSCAR_initial", self.atoms, "vasp")
        dyn.run(self.params.ofdft_number_of_timesteps)
        ase.io.write("POSCAR_equilibrated", self.atoms, "vasp")

