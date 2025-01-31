"""Tools for initializing a (ML)-DFT trajectory with OF-DFT."""

from warnings import warn

from ase import units
import ase.io
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

try:
    from dftpy.api.api4ase import DFTpyCalculator
    from dftpy.config import DefaultOption, OptionFormat
except ModuleNotFoundError:
    pass


class OFDFTInitializer:
    """
    Initializes a trajectory using OF-DFT.

    Parameters
    ----------
    parameters : mala.Parameters
        MALA parameters object used to create this instance.

    atoms : ase.Atoms
        Initial atomic configuration for which an equilibrated configuration
        is to be created.


    Attributes
    ----------
    parameters : mala.mala.common.parameters.ParametersDataGeneration
        MALA data generation parameters object.

    atoms : ase.Atoms
        Initial atomic configuration for which an
        equilibrated configuration is to be created.

    dftpy_configuration : dict
        Dictionary containing the DFTpy configuration. Will partially be
        populated via the MALA parameters object.
    """

    def __init__(self, parameters, atoms):
        warn(
            "The class OFDFTInitializer is experimental. The algorithms "
            "within have been tested, but the API may still be subject to "
            "large changes."
        )
        self.atoms = atoms
        self.parameters = parameters.datageneration

        # Check that only one element is used in the atoms.
        number_of_elements = len(set([x.symbol for x in self.atoms]))
        if number_of_elements > 1:
            raise Exception(
                "OF-DFT-MD initialization can only work with one element."
            )
        self.dftpy_configuration = DefaultOption()

        self.dftpy_configuration["PATH"][
            "pppath"
        ] = self.parameters.local_psp_path
        self.dftpy_configuration["PP"][
            self.atoms[0].symbol
        ] = self.parameters.local_psp_name
        self.dftpy_configuration["OPT"]["method"] = self.parameters.ofdft_kedf
        self.dftpy_configuration["KEDF"]["kedf"] = "WT"
        self.dftpy_configuration["JOB"]["calctype"] = "Energy Force"

    def get_equilibrated_configuration(self, logging_period=None):
        """
        Calculate the (OF-DFT-MD) equilibrated atomic configuration.

        Parameters
        ----------
        logging_period : int
            If not None, a .log and .traj file will be filled with snapshot
            information every logging_period steps.

        Returns
        -------
        equilibrated_configuration : ase.Atoms
            Equilibrated atomic configuration.
        """
        # Set the DFTPy configuration.
        conf = OptionFormat(self.dftpy_configuration)

        # Create the DFTPy Calculator.
        calc = DFTpyCalculator(config=conf)
        self.atoms.set_calculator(calc)

        # Create the initial velocities, and dynamics object.
        MaxwellBoltzmannDistribution(
            self.atoms,
            temperature_K=self.parameters.ofdft_temperature,
            force_temp=True,
        )
        dyn = Langevin(
            self.atoms,
            self.parameters.ofdft_timestep * units.fs,
            temperature_K=self.parameters.ofdft_temperature,
            friction=self.parameters.ofdft_friction,
        )

        # If logging is desired, do the logging.
        if logging_period is not None:
            dyn.attach(
                MDLogger(
                    dyn,
                    self.atoms,
                    "mala_of_dft_md.log",
                    header=False,
                    stress=False,
                    peratom=True,
                    mode="w",
                ),
                interval=logging_period,
            )
            traj = Trajectory("mala_of_dft_md.traj", "w", self.atoms)

            dyn.attach(traj.write, interval=logging_period)

        # Let the OF-DFT-MD run.
        ase.io.write("POSCAR_initial", self.atoms, "vasp")
        dyn.run(self.parameters.ofdft_number_of_timesteps)
        ase.io.write("POSCAR_equilibrated", self.atoms, "vasp")
