"""
Framework for electronic structure learning.

Can be used to preprocess DFT data (positions / LDOS), train networks,
predict LDOS and postprocess LDOS into energies (and forces, soon).
"""

from .version import __version__
from .common import Parameters, printout, check_modules
from .descriptors import SNAP, Descriptor
from .datahandling import DataHandler, DataScaler, DataConverter, Snapshot
from .network import Network, Tester, Trainer, HyperOpt, \
    HyperOptOptuna, HyperOptNASWOT, HyperOptOAT, Predictor, \
    HyperparameterOAT, HyperparameterNASWOT, HyperparameterOptuna
from .targets import LDOS, DOS, Density, fermi_function, \
    AtomicForce, Target
from .interfaces import MALA
from .datageneration import TrajectoryAnalyzer, OFDFTInitializer
