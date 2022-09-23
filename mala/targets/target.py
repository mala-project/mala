"""Base class for all target calculators."""
from abc import ABC, abstractmethod
import itertools

from ase.neighborlist import NeighborList
from ase.units import Rydberg, kB
import ase.io
import numpy as np
from scipy.spatial import distance
from scipy.integrate import simps

from mala.common.parameters import Parameters, ParametersTargets
from mala.common.parallelizer import printout, parallel_warn
from mala.targets.calculation_helpers import fermi_function

# Empirical value for the Gaussian descriptor width, determined for an
# aluminium system. Reasonable values for sigma can and will be calculated
# automatically based on this value and the aluminium gridspacing
# for other systems as well.
optimal_sigma_aluminium = 0.2
reference_grid_spacing_aluminium = 0.08099000022712448


class Target(ABC):
    """
    Base class for all target quantity parser.

    Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file
    format and performs postprocessing calculations on the quantity.

    Parameters
    ----------
    params : mala.common.parameters.Parameters or
    mala.common.parameters.ParametersTargets
        Parameters used to create this Target object.
    """

    def __new__(cls, params: Parameters):
        """
        Create a Target instance.

        The correct type of target calculator will automatically be
        instantiated by this class if possible. You can also instantiate
        the desired target directly by calling upon the subclass.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this target calculator.
        """
        target = None

        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == Target:
            if isinstance(params, Parameters):
                targettype = params.targets.target_type
            elif isinstance(params, ParametersTargets):
                targettype = params.target_type
            else:
                raise Exception("Wrong type of parameters for Targets class.")

            if targettype == 'LDOS':
                from mala.targets.ldos import LDOS
                target = super(Target, LDOS).__new__(LDOS)
            if targettype == 'DOS':
                from mala.targets.dos import DOS
                target = super(Target, DOS).__new__(DOS)
            if targettype == 'Density':
                from mala.targets.density import Density
                target = super(Target, Density).__new__(Density)

            if target is None:
                raise Exception("Unsupported target parser/calculator.")
        else:
            target = super(Target, cls).__new__(cls)

        return target

    def __init__(self, params):
        self._parameters_full = None
        if isinstance(params, Parameters):
            self.parameters: ParametersTargets = params.targets
            self._parameters_full = params
        elif isinstance(params, ParametersTargets):
            self.parameters: ParametersTargets = params
        else:
            raise Exception("Wrong type of parameters for Targets class.")
        self.fermi_energy_dft = None
        self.temperature = None
        self.voxel = None
        self.number_of_electrons_exact = None
        self.number_of_electrons_from_eigenvals = None
        self.band_energy_dft_calculation = None
        self.total_energy_dft_calculation = None
        self.grid_dimensions = [0, 0, 0]
        self.atoms = None
        self.electrons_per_atom = None
        self.qe_input_data = {
                "occupations": 'smearing',
                "calculation": 'scf',
                "restart_mode": 'from_scratch',
                "prefix": 'MALA',
                "pseudo_dir": self.parameters.pseudopotential_path,
                "outdir": './',
                "ibrav": None,
                "smearing": 'fermi-dirac',
                "degauss": None,
                "ecutrho": None,
                "ecutwfc": None,
                "nosym": True,
                "noinv": True,
        }

        # It has been shown that the number of k-points
        # does not really affect the QE post-processing calculation.
        # This is because we only evaulate density-dependent contributions
        # with QE. However, there were some (very small) inaccuracies when
        # operating only at the gamma point. Consequently, MALA defaults
        # to a small k-grid to ensure best accuracy and performance.
        # UPDATE 23.04.2021: As per discussion bewteen Normand Modine and Lenz
        # Fiedler, the number of k-points is moved to 1.
        # The small inaccuracies are neglected for now.
        self.kpoints = None  # (2, 2, 2)
        self.qe_pseudopotentials = {}

        # Local grid and parallelization info for distributed inference.
        self.local_grid = None
        self.y_planes = None

    @property
    @abstractmethod
    def feature_size(self):
        """Get dimension of this target if used as feature in ML."""
        pass

    @property
    def qe_input_data(self):
        """Input data for QE TEM calls."""
        # Update the pseudopotential path from Parameters.
        self._qe_input_data["pseudo_dir"] = \
            self.parameters.pseudopotential_path
        return self._qe_input_data

    @qe_input_data.setter
    def qe_input_data(self, value):
        self._qe_input_data = value

    def _is_property_cached(self, property_name):
        return property_name in self.__dict__.keys()

    @abstractmethod
    def get_target(self):
        """
        Get the target quantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        pass

    @abstractmethod
    def invalidate_target(self):
        """
        Invalidates the saved target wuantity.

        This is the generic interface for cached target quantities.
        It should work for all implemented targets.
        """
        pass

    def read_additional_calculation_data(self, data_type, data=""):
        """
        Read in additional input about a calculation.

        This is e.g. necessary when we operate with preprocessed
        data for the training itself but want to take into account other
        physical quantities (such as the fermi energy or the electronic
        temperature) for post processing.

        Parameters
        ----------
        data_type : string
            Type of data or file that is used. Currently supporte are:

            - "qe.out" : Read the additional information from a QuantumESPRESSO
              output file.

            - "atoms+grid" : Provide a grid and an atoms object from which to
              predict. Except for the number of electrons,
              this mode will not change any member variables;
              values have to be adjusted BEFORE.

        data : string or List
            Data from which additional calculation data is inputted.
        """
        if data_type == "qe.out":
            # Reset everything.
            self.fermi_energy_dft = None
            self.temperature = None
            self.number_of_electrons_exact = None
            self.voxel = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms = None

            # Read the file.
            self.atoms = ase.io.read(data, format="espresso-out")
            vol = self.atoms.get_volume()
            self.fermi_energy_dft = self.atoms.get_calculator().\
                get_fermi_level()

            # Parse the file for energy values.
            total_energy = None
            past_calculation_part = False
            bands_included = True
            with open(data) as out:
                pseudolinefound = False
                lastpseudo = None
                for line in out:
                    if "End of self-consistent calculation" in line:
                        past_calculation_part = True
                    if "number of electrons       =" in line:
                        self.number_of_electrons_exact = \
                            np.float64(line.split('=')[1])
                    if "Fermi-Dirac smearing, width (Ry)=" in line:
                        self.temperature = np.float64(line.split('=')[2]) * \
                                           Rydberg / kB
                    if "xc contribution" in line:
                        break
                    if "FFT dimensions" in line:
                        dims = line.split("(")[1]
                        self.grid_dimensions[0] = int(dims.split(",")[0])
                        self.grid_dimensions[1] = int(dims.split(",")[1])
                        self.grid_dimensions[2] = int((dims.split(",")[2]).
                                                      split(")")[0])
                    if "bravais-lattice index" in line:
                        self.qe_input_data["ibrav"] = int(line.split("=")[1])
                    if "kinetic-energy cutoff" in line:
                        self.qe_input_data["ecutwfc"] \
                            = float((line.split("=")[1]).split("Ry")[0])
                    if "charge density cutoff" in line:
                        self.qe_input_data["ecutrho"] \
                            = float((line.split("=")[1]).split("Ry")[0])
                    if "smearing, width" in line:
                        self.qe_input_data["degauss"] \
                            = float(line.split("=")[-1])
                    if pseudolinefound:
                        self.qe_pseudopotentials[lastpseudo.strip()] \
                            = line.split("/")[-1].strip()
                        pseudolinefound = False
                        lastpseudo = None
                    if "PseudoPot." in line:
                        pseudolinefound = True
                        lastpseudo = (line.split("for")[1]).split("read")[0]
                    if "total energy" in line and past_calculation_part:
                        if total_energy is None:
                            total_energy \
                                = float((line.split('=')[1]).split('Ry')[0])
                    if "set verbosity='high' to print them." in line:
                        bands_included = False

            # The voxel is needed for e.g. LDOS integration.
            self.voxel = self.atoms.cell.copy()
            self.voxel[0] = self.voxel[0] / (
                        self.grid_dimensions[0])
            self.voxel[1] = self.voxel[1] / (
                        self.grid_dimensions[1])
            self.voxel[2] = self.voxel[2] / (
                        self.grid_dimensions[2])
            self._parameters_full.descriptors.gaussian_descriptors_sigma = \
                (np.max(self.voxel) / reference_grid_spacing_aluminium) * \
                optimal_sigma_aluminium

            # This is especially important for size extrapolation.
            self.electrons_per_atom = self.number_of_electrons_exact / \
                len(self.atoms)

            # Unit conversion
            self.total_energy_dft_calculation = total_energy*Rydberg

            # Calculate band energy, if the necessary data is included in
            # the output file.
            if bands_included:
                eigs = np.transpose(
                    self.atoms.get_calculator().band_structure().
                    energies[0, :, :])
                kweights = self.atoms.get_calculator().get_k_point_weights()
                eband_per_band = eigs * fermi_function(eigs,
                                                       self.fermi_energy_dft,
                                                       self.temperature)
                eband_per_band = kweights[np.newaxis, :] * eband_per_band
                self.band_energy_dft_calculation = np.sum(eband_per_band)
                enum_per_band = fermi_function(eigs, self.fermi_energy_dft,
                                               self.temperature)
                enum_per_band = kweights[np.newaxis, :] * enum_per_band
                self.number_of_electrons_from_eigenvals = np.sum(enum_per_band)

        elif data_type == "atoms+grid":
            # Reset everything that we can get this way.
            self.voxel = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms: ase.Atoms = data[0]

            # Read the file.
            vol = self.atoms.get_volume()

            # Parse the file for energy values.
            self.grid_dimensions[0] = data[1][0]
            self.grid_dimensions[1] = data[1][1]
            self.grid_dimensions[2] = data[1][2]

            # The voxel is needed for e.g. LDOS integration.
            self.voxel = self.atoms.cell.copy()
            self.voxel[0] = self.voxel[0] / (
                        self.grid_dimensions[0])
            self.voxel[1] = self.voxel[1] / (
                        self.grid_dimensions[1])
            self.voxel[2] = self.voxel[2] / (
                        self.grid_dimensions[2])
            self._parameters_full.descriptors.gaussian_descriptors_sigma = \
                (np.max(self.voxel) / reference_grid_spacing_aluminium) * \
                optimal_sigma_aluminium

            if self.electrons_per_atom is None:
                printout("No number of electrons per atom provided, "
                         "MALA cannot guess the number of electrons "
                         "in the cell with this. Energy calculations may be"
                         "wrong.")

            else:
                self.number_of_electrons_exact = self.electrons_per_atom * \
                                                 len(self.atoms)

        else:
            raise Exception("Unsupported auxiliary file type.")

    def get_energy_grid(self):
        """Get energy grid."""
        raise Exception("No method implement to calculate an energy grid.")

    def get_real_space_grid(self):
        """Get the real space grid."""
        grid3D = np.zeros((self.grid_dimensions[0], self.grid_dimensions[1],
                           self.grid_dimensions[2], 3), dtype=np.float64)
        for i in range(0, self.grid_dimensions[0]):
            for j in range(0, self.grid_dimensions[1]):
                for k in range(0, self.grid_dimensions[2]):
                    grid3D[i, j, k, :] = np.matmul(self.voxel, [i, j, k])
        return grid3D

    @staticmethod
    def _get_ideal_rmax_for_rdf(atoms: ase.Atoms, method="mic"):
        if method == "mic":
            return np.min(np.linalg.norm(atoms.get_cell(), axis=0))/2
        elif method == "2mic":
            return np.min(np.linalg.norm(atoms.get_cell(), axis=0)) - 0.0001
        else:
            raise Exception("Unknown option to calculate rMax provided.")

    @staticmethod
    def radial_distribution_function_from_atoms(atoms: ase.Atoms,
                                                number_of_bins,
                                                rMax="mic",
                                                method="mala"):
        """
        Calculate the radial distribution function (RDF).

        In comparison with other python implementation, this function
        can handle arbitrary radii by incorporating a neighbor list.
        Loosely based on the implementation in the ASAP3 code
        (specifically, this file:
        https://gitlab.com/asap/asap/-/blob/master/Tools/RawRadialDistribution.cpp),
        but without the restriction for the radii.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        number_of_bins : int
            Number of bins used to create the histogram.

        rMax : float or string
            Radius up to which to calculate the RDF.
            Options are:

                - "mic": rMax according to minimum image convention.
                  (see "The Minimum Image Convention in Non-Cubic MD Cells"
                  by Smith,
                  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696)
                  Suggested value, as this leads to physically meaningful RDFs.
                - "2mic": rMax twice as big as for "mic". Legacy option
                  to reproduce some earlier trajectory results
                - float: Input some float to use it directly as input.

        method : string
            If "mala" the more flexible, yet less performant python based
            RDF will be calculated. If "asap3", asap3's C++ implementation
            will be used. In the latter case, rMax values larger then
            "2mic" will be ignored.

        Returns
        -------
        rdf : numpy.ndarray
            The RDF as [rMax] array.

        radii : numpy.ndarray
            The radii  at which the RDF was calculated (for plotting),
            as [rMax] array.
        """
        # Leaving this here for potential future use - this is much faster
        # because C++.
        # from asap3.analysis.rdf import RadialDistributionFunction
        # (RadialDistributionFunction(atoms, rMax, 500)).get_rdf()

        if rMax == "mic":
            _rMax = Target._get_ideal_rmax_for_rdf(atoms, method="mic")
        elif rMax == "2mic":
            _rMax = Target._get_ideal_rmax_for_rdf(atoms, method="2mic")
        else:
            if method == "asap3":
                _rMax_possible = Target._get_ideal_rmax_for_rdf(atoms,
                                                                method="2mic")
                if rMax > _rMax_possible:
                    raise Exception("ASAP3 calculation fo RDF cannot work "
                                    "with radii that are bigger then the "
                                    "cell.")
            _rMax = rMax

        atoms = atoms
        dr = float(_rMax / number_of_bins)

        if method == "mala":
            rdf = np.zeros(number_of_bins + 1)

            cell = atoms.get_cell()
            pbc = atoms.get_pbc()
            for i in range(0, 3):
                if pbc[i]:
                    if _rMax > cell[i, i]:
                        parallel_warn(
                            "Calculating RDF with a radius larger then the "
                            "unit cell. While this will work numerically, be "
                            "cautious about the physicality of its results")

            # Calculate all the distances.
            # rMax/2 because this is the radius around one atom, so half the
            # distance to the next one.
            # Using neighborlists grants us access to the PBC.
            neighborlist = ase.neighborlist.NeighborList(np.zeros(len(atoms)) +
                                                         [_rMax/2.0],
                                                         bothways=True)
            neighborlist.update(atoms)
            for i in range(0, len(atoms)):
                indices, offsets = neighborlist.get_neighbors(i)
                dm = distance.cdist([atoms.get_positions()[i]],
                                    atoms.positions[indices] + offsets @
                                    atoms.get_cell())
                index = (np.ceil(dm / dr)).astype(int)
                index = index.flatten()
                out_of_scope = index > number_of_bins
                index[out_of_scope] = 0
                for idx in index:
                    rdf[idx] += 1

            # Normalize the RDF and calculate the distances
            rr = []
            phi = len(atoms) / atoms.get_volume()
            norm = 4.0 * np.pi * dr * phi * len(atoms)
            for i in range(1, number_of_bins + 1):
                rr.append((i - 0.5) * dr)
                rdf[i] /= (norm * ((rr[-1] ** 2) + (dr ** 2) / 12.))
        elif method == "asap3":
            # ASAP3 loads MPI which takes a long time to import, so
            # we'll only do that when absolutely needed.
            from asap3.analysis.rdf import RadialDistributionFunction
            rdf = RadialDistributionFunction(atoms, _rMax,
                                             number_of_bins).get_rdf()
            rr = []
            for i in range(1, number_of_bins + 1):
                rr.append((i - 0.5) * dr)
            return rdf, rr
        else:
            raise Exception("Unknown RDF method selected.")
        return rdf[1:], rr
        
    @staticmethod
    def three_particle_correlation_function_from_atoms(atoms: ase.Atoms,
                                                       number_of_bins,
                                                       rMax="mic"):
        """
        Calculate the three particle correlation function (TPCF).

        The implementation was done as given by doi.org/10.1063/5.0048450,
        equation 17, with only small modifications. Pleas be aware that,
        while optimized, this function tends to be compuational heavy for
        large radii or number of bins.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        number_of_bins : int
            Number of bins used to create the histogram.

        rMax : float or string
            Radius up to which to calculate the RDF.
            Options are:

                - "mic": rMax according to minimum image convention.
                  (see "The Minimum Image Convention in Non-Cubic MD Cells"
                  by Smith,
                  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696)
                  Suggested value, as this leads to physically meaningful RDFs.
                - "2mic : rMax twice as big as for "mic". Legacy option
                  to reproduce some earlier trajectory results
                - float: Input some float to use it directly as input.

        Returns
        -------
        tpcf : numpy.ndarray
            The TPCF as [rMax, rMax, rMax] array.

        radii : numpy.ndarray
            The radii at which the TPCF was calculated (for plotting),
            [rMax, rMax, rMax].
        """
        if rMax == "mic":
            _rMax = Target._get_ideal_rmax_for_rdf(atoms, method="mic")
        elif rMax == "2mic":
            _rMax = Target._get_ideal_rmax_for_rdf(atoms, method="2mic")
        else:
            _rMax = rMax

        # TPCF is a function of three radii.
        atoms = atoms
        dr = float(_rMax/number_of_bins)
        tpcf = np.zeros([number_of_bins + 1, number_of_bins + 1,
                        number_of_bins + 1])
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        for i in range(0, 3):
            if pbc[i]:
                if _rMax > cell[i, i]:
                    raise Exception("Cannot calculate RDF with this radius. "
                                    "Please choose a smaller value.")

        # Construct a neighbor list for calculation of distances.
        # With this, the PBC are satisfied.
        neighborlist = ase.neighborlist.NeighborList(np.zeros(len(atoms)) +
                                                     [_rMax/2.0],
                                                     bothways=True)
        neighborlist.update(atoms)

        # To calculate the TPCF we calculate the three distances between
        # three atoms. We use a neighbor list for this.
        for i in range(0, len(atoms)):

            # Debugging
            # if i == 1:
            #     break

            pos1 = atoms.get_positions()[i]
            # Generate all pairs of atoms, and calculate distances of
            # reference atom to them.
            indices, offsets = neighborlist.get_neighbors(i)
            neighbor_pairs = itertools.\
                combinations(list(zip(indices, offsets)), r=2)
            neighbor_list = list(neighbor_pairs)
            pair_positions = np.array([np.concatenate((atoms.positions[pair1[0]] + \
                                                       pair1[1] @ atoms.get_cell(),
                                                       atoms.positions[pair2[0]] + \
                                                       pair2[1] @ atoms.get_cell()))
                                       for pair1, pair2 in neighbor_list])
            dists_between_atoms = np.sqrt(
                np.square(pair_positions[:, 0] - pair_positions[:, 3]) +
                np.square(pair_positions[:, 1] - pair_positions[:, 4]) +
                np.square(pair_positions[:, 2] - pair_positions[:, 5]))
            pair_positions = np.reshape(pair_positions, (len(neighbor_list)*2,
                                                         3), order="C")
            all_dists = distance.cdist([pos1], pair_positions)[0]

            for idx, neighbor_pair in enumerate(neighbor_list):
                r1 = all_dists[2*idx]
                r2 = all_dists[2*idx+1]

                # We don't need to do any calculation if either of the
                # atoms are already out of range.
                if r1 < _rMax and r2 < _rMax:
                    r3 = dists_between_atoms[idx]
                    if r3 < _rMax and np.abs(r1-r2) < r3 < (r1+r2):
                        # print(r1, r2, r3)
                        id1 = (np.ceil(r1 / dr)).astype(int)
                        id2 = (np.ceil(r2 / dr)).astype(int)
                        id3 = (np.ceil(r3 / dr)).astype(int)
                        tpcf[id1, id2, id3] += 1

        # Normalize the TPCF and calculate the distances.
        # This loop takes almost no time compared to the one above.
        rr = np.zeros([3, number_of_bins+1, number_of_bins+1,
                       number_of_bins+1])
        phi = len(atoms) / atoms.get_volume()
        norm = 8.0 * np.pi * np.pi * dr * phi * phi * len(atoms)
        for i in range(1, number_of_bins + 1):
            for j in range(1, number_of_bins + 1):
                for k in range(1, number_of_bins + 1):
                    r1 = (i - 0.5) * dr
                    r2 = (j - 0.5) * dr
                    r3 = (k - 0.5) * dr
                    tpcf[i, j, k] /= (norm * r1 * r2 * r3
                                      * dr * dr * dr)
                    rr[0, i, j, k] = r1
                    rr[1, i, j, k] = r2
                    rr[2, i, j, k] = r3
        return tpcf[1:, 1:, 1:], rr[:, 1:, 1:, 1:]

    @staticmethod
    def static_structure_factor_from_atoms(atoms: ase.Atoms, number_of_bins,
                                           kMax,
                                           radial_distribution_function=None,
                                           calculation_type="direct"):
        """
        Calculate the static structure factor (SSF).

        Implemented according to https://arxiv.org/abs/1606.03610:

            - Eq. 2 as calculation_type = "direct", i.e. via summation in Fourier space
            - Eq. 6 as calculation_type = "fourier_transform", i.e. via calculation of RDF and thereafter Fourier transformation.

        The direct calculation is significantly more accurate and about
        as efficient, at least for small systems.
        In either case, the SSF will be given as S(k), even though
        technically, the direct method calculates S(**k**); during binning,
        this function averages over all **k** with the same k.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        number_of_bins : int
            Number of bins used to create the histogram.

        kMax : float
            Maximum wave vector up to which to calculate the SSF.

        radial_distribution_function : List of numpy.ndarray
            If not None, and calculation_type="fourier_transform", this RDf
            will be used for the Fourier transformation.

        calculation_type : string
            Either "direct" or "fourier_transform". Controls how the SSF is
            calculated.

        Returns
        -------
        ssf : numpy.ndarray
            The SSF as [kMax] array.

        kpoints : numpy.ndarray
            The k-points  at which the SSF was calculated (for plotting),
            as [kMax] array.
        """
        if calculation_type == "fourier_transform":
            if radial_distribution_function is None:
                rMax = Target._get_ideal_rmax_for_rdf(atoms)*6
                radial_distribution_function = Target.\
                    radial_distribution_function_from_atoms(atoms, rMax=rMax,
                                                            number_of_bins=
                                                            1500)
            rdf = radial_distribution_function[0]
            radii = radial_distribution_function[1]

            structure_factor = np.zeros(number_of_bins + 1)
            dk = float(kMax / number_of_bins)
            kpoints = []

            # Fourier transform the RDF by calculating the integral at each
            # k-point we investigate.
            rho = len(atoms)/atoms.get_volume()
            for i in range(0, number_of_bins + 1):
                # Construct integrand.
                kpoints.append(dk*i)
                kr = np.array(radii)*kpoints[-1]
                integrand = (rdf-1)*radii*np.sin(kr)/kpoints[-1]
                structure_factor[i] = 1 + (4*np.pi*rho * simps(integrand,
                                                               radii))

            return structure_factor[1:], np.array(kpoints)[1:]

        elif calculation_type == "direct":
            # This is the delta for the binning.
            # The delta for the generation of the k-points themselves is
            # fixed as delta k = 2pi/L with L being the length of a
            # cubic cell. More generally, it can be determined via the
            # reciprocal unit vectors.
            # The structure factor is undefined for wave vectors smaller
            # then this number.
            dk = float(kMax / number_of_bins)
            dk_threedimensional = atoms.get_cell().reciprocal()*2*np.pi

            # From this, the necessary dimensions of the k-grid for this
            # particular k-max can be determined as
            kgrid_size = np.ceil(np.matmul(np.linalg.inv(dk_threedimensional),
                                           [kMax, kMax, kMax])).astype(int)
            print("Calculating SSF on k-grid of size", kgrid_size)

            # k-grids:
            # The first will hold the full 3D k-points, the second only
            # |k|.
            kgrid = []
            for i in range(0, kgrid_size[0]):
                for j in range(0, kgrid_size[1]):
                    for k in range(0, kgrid_size[2]):
                        k_point = np.matmul(dk_threedimensional, [i, j, k])
                        if np.linalg.norm(k_point) <= kMax:
                            kgrid.append(k_point)
            kpoints = []
            for i in range(0, number_of_bins + 1):
                kpoints.append(dk*i)

            # The first will hold S(|k|) (i.e., what we are actually interested
            # in, the second will hold lists of all S(k) corresponding to the
            # same |k|.
            structure_factor = np.zeros(number_of_bins + 1)
            structure_factor_kpoints = []
            for i in range(0, number_of_bins + 1):
                structure_factor_kpoints.append([])

            # Dot product, cosine and sine calculation are done as vector
            # operations. I am aware this is not particularly memory-friendly,
            # but it is fast.
            dot_product = np.dot(kgrid, np.transpose(atoms.get_positions()))
            cosine_sum = np.sum(np.cos(dot_product), axis=1)
            sine_sum = np.sum(np.sin(dot_product), axis=1)
            del dot_product
            s_values = (np.square(cosine_sum)+np.square(sine_sum)) / len(atoms)
            del cosine_sum
            del sine_sum

            # s_value holds S(k), now we have to group this into
            # S(|k|).
            k_value = np.linalg.norm(kgrid, axis=1)
            indices = (np.ceil(k_value / dk)).astype(int)
            indices = indices.flatten()
            for idx, index in enumerate(indices):
                structure_factor_kpoints[index].append(s_values[idx])

            for i in range(1, number_of_bins + 1):
                if len(structure_factor_kpoints[i]) > 0:
                    structure_factor[i] = np.mean(structure_factor_kpoints[i])

            return structure_factor[1:], np.array(kpoints)[1:]

        else:
            raise Exception("Static structure factor calculation method "
                            "unsupported.")

    def get_radial_distribution_function(self, atoms: ase.Atoms,
                                         method="mala"):
        """
        Calculate the radial distribution function (RDF).

        Uses the values as given in the parameters object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        method : string
            If "mala" the more flexible, yet less performant python based
            RDF will be calculated. If "asap3", asap3's C++ implementation
            will be used.

        Returns
        -------
        rdf : numpy.ndarray
            The RDF as [rMax] array.

        radii : numpy.ndarray
            The radii  at which the RDF was calculated (for plotting),
            as [rMax] array.

        method : string
            If "mala" the more flexible, yet less performant python based
            RDF will be calculated. If "asap3", asap3's C++ implementation
            will be used. In the latter case, rMax will be ignored and
            automatically calculated.

        """
        return Target.\
            radial_distribution_function_from_atoms(atoms,
                                                    number_of_bins=self.
                                                    parameters.
                                                    rdf_parameters
                                                    ["number_of_bins"],
                                                    rMax=self.parameters.
                                                    rdf_parameters["rMax"],
                                                    method=method)

    def get_three_particle_correlation_function(self, atoms: ase.Atoms):
        """
        Calculate the three particle correlation function (TPCF).

        Uses the values as given in the parameters object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        Returns
        -------
        tpcf : numpy.ndarray
            The TPCF as [rMax, rMax, rMax] array.

        radii : numpy.ndarray
            The radii at which the TPCF was calculated (for plotting),
            [rMax, rMax, rMax].
        """
        return Target.\
            three_particle_correlation_function_from_atoms(atoms,
                                                           number_of_bins=self.
                                                           parameters.
                                                           tpcf_parameters
                                                           ["number_of_bins"],
                                                           rMax=self.parameters.
                                                           tpcf_parameters["rMax"])

    def get_static_structure_factor(self, atoms: ase.Atoms):
        """
        Calculate the static structure factor (SSF).

        Uses the values as given in the parameters object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms for which to construct the RDF.

        Returns
        -------
        ssf : numpy.ndarray
            The SSF as [kMax] array.

        kpoints : numpy.ndarray
            The k-points  at which the SSF was calculated (for plotting),
            as [kMax] array.
        """
        return Target.static_structure_factor_from_atoms(atoms,
                                                         self.parameters.
                                                         ssf_parameters["number_of_bins"],
                                                         self.parameters.
                                                         ssf_parameters["number_of_bins"])

    @staticmethod
    def write_tem_input_file(atoms_Angstrom, qe_input_data,
                             qe_pseudopotentials,
                             grid_dimensions, kpoints):
        """
        Write a QE-style input file for the total energy module.

        Usually, the used parameters should correspond to the properties of
        the object calling this function, but they don't necessarily have
        to.

        Parameters
        ----------
        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        grid_dimensions : List
            A list containing the x,y,z dimensions of the real space grid.

        kpoints : dict
            k-grid used, usually None or (1,1,1) for TEM calculations.
        """
        # Specify grid dimensions, if any are given.
        if grid_dimensions[0] != 0 and \
                grid_dimensions[1] != 0 and \
                grid_dimensions[2] != 0:
            qe_input_data["nr1"] = grid_dimensions[0]
            qe_input_data["nr2"] = grid_dimensions[1]
            qe_input_data["nr3"] = grid_dimensions[2]
            qe_input_data["nr1s"] = grid_dimensions[0]
            qe_input_data["nr2s"] = grid_dimensions[1]
            qe_input_data["nr3s"] = grid_dimensions[2]

        # Might be needed for test purposes, the Be2 test data
        # for example has symmetry, even though it was deactivated for
        # the DFT calculation. If symmetry is then on in here, that
        # leads to errors.
        # qe_input_data["nosym"] = False
        ase.io.write("mala.pw.scf.in", atoms_Angstrom, "espresso-in",
                     input_data=qe_input_data,
                     pseudopotentials=qe_pseudopotentials,
                     kpts=kpoints)

    @staticmethod
    def convert_units(array, in_units="eV"):
        """
        Convert the units of an array into the MALA units.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.

        """
        raise Exception("No unit conversion method implemented for"
                        " this target type.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from MALA units into desired units.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.

        """
        raise Exception("No unit back conversion method implemented "
                        "for this target type.")

    def restrict_data(self, array):
        """
        Restrict target data to only contain physically meaningful values.

        For the LDOS this e.g. implies non-negative values. The type
        of data restriction is specified by the parameters.

        Parameters
        ----------
        array : numpy.array
            Numpy array, for which the restrictions are to be applied.

        Returns
        -------
        array : numpy.array
            The same array, with restrictions enforced.
        """
        if self.parameters.restrict_targets == "zero_out_negative":
            array[array < 0] = 0
            return array
        elif self.parameters.restrict_targets == "absolute_values":
            array[array < 0] *= -1
            return array
        elif self.parameters.restrict_targets is None:
            return array
        else:
            raise Exception("Wrong data restriction.")


