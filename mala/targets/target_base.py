"""Base class for all target calculators."""
from abc import ABC, abstractmethod
from copy import deepcopy
import itertools

from ase.neighborlist import NeighborList
from ase.units import Rydberg, Bohr, kB
from ase.dft.kpoints import monkhorst_pack
import ase.io
import numpy as np
from scipy.spatial import distance

from mala.common.parameters import Parameters, ParametersTargets
from mala.targets.calculation_helpers import fermi_function


class TargetBase(ABC):
    """
    Base class for all target quantity parser.

    Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file
    format and performs postprocessing calculations on the quantity.

    Parameters
    ----------
    params : mala.common.parameters.Parameters or
    mala.common.parameters.ParametersTargets
        Parameters used to create this TargetBase object.
    """

    def __init__(self, params):
        if isinstance(params, Parameters):
            self.parameters = params.targets
        elif isinstance(params, ParametersTargets):
            self.parameters = params
        else:
            raise Exception("Wrong type of parameters for Targets class.")
        self.fermi_energy_eV = None
        self.temperature_K = None
        self.grid_spacing_Bohr = None
        self.number_of_electrons = None
        self.number_of_electrons_from_eigenvals = None
        self.band_energy_dft_calculation = None
        self.total_energy_dft_calculation = None
        self.grid_dimensions = [0, 0, 0]
        self.atoms = None
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

    @abstractmethod
    def get_feature_size(self):
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

    def read_from_cube(self):
        """Read the quantity from a .cube file."""
        raise Exception("No function defined to read this quantity "
                        "from a .cube file.")

    def read_from_qe_dos_txt(self):
        """Read the quantity from a Quantum Espresso .dos.txt file."""
        raise Exception("No function defined to read this quantity "
                        "from a qe.dos.txt file")

    def get_density(self):
        """Get the electronic density."""
        raise Exception("No function to calculate or provide the "
                        "density has been implemented for this target type.")

    def get_density_of_states(self):
        """Get the density of states."""
        raise Exception("No function to calculate or provide the"
                        "density of states (DOS) has been implemented "
                        "for this target type.")

    def get_band_energy(self):
        """Get the band energy."""
        raise Exception("No function to calculate or provide the"
                        "band energy has been implemented for this target "
                        "type.")

    def get_number_of_electrons(self):
        """Get the number of electrons."""
        raise Exception("No function to calculate or provide the number of"
                        " electrons has been implemented for this target "
                        "type.")

    def get_total_energy(self):
        """Get the total energy."""
        raise Exception("No function to calculate or provide the number "
                        "of electons has been implemented for this target "
                        "type.")

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
            Type of data or file that is used. Currently only supports
            qe.out for Quantum Espresso outfiles.

        data : string or list
            Data from which additional calculation data is inputted.
        """
        if data_type == "qe.out":
            # Reset everything.
            self.fermi_energy_eV = None
            self.temperature_K = None
            self.grid_spacing_Bohr = None
            self.number_of_electrons = None
            self.band_energy_dft_calculation = None
            self.total_energy_dft_calculation = None
            self.grid_dimensions = [0, 0, 0]
            self.atoms = None

            # Read the file.
            self.atoms = ase.io.read(data, format="espresso-out")
            vol = self.atoms.get_volume()
            self.fermi_energy_eV = self.atoms.get_calculator().\
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
                        self.number_of_electrons = np.float64(line.split('=')
                                                              [1])
                    if "Fermi-Dirac smearing, width (Ry)=" in line:
                        self.temperature_K = np.float64(line.split('=')[2]) * \
                                             Rydberg / kB
                    if "xc contribution" in line:
                        xc_contribution = float((line.split('=')[1]).
                                                split('Ry')[0])
                        break
                    if "one-electron contribution" in line:
                        one_electron_contribution = float((line.split('=')[1]).
                                                          split('Ry')[0])
                    if "hartree contribution" in line:
                        hartree_contribution = float((line.split('=')[1]).
                                                     split('Ry')[0])
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

            # Post process the text values.
            cell_volume = vol / (self.grid_dimensions[0] *
                                 self.grid_dimensions[1] *
                                 self.grid_dimensions[2] * Bohr ** 3)
            self.grid_spacing_Bohr = cell_volume ** (1 / 3)

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
                                                       self.fermi_energy_eV,
                                                       self.temperature_K)
                eband_per_band = kweights[np.newaxis, :] * eband_per_band
                self.band_energy_dft_calculation = np.sum(eband_per_band)
                enum_per_band = fermi_function(eigs,
                                               self.fermi_energy_eV,
                                               self.temperature_K)
                enum_per_band = kweights[np.newaxis, :] * enum_per_band
                self.number_of_electrons_from_eigenvals = np.sum(enum_per_band)
        elif data_type == "atoms+grid":
            # Reset everything that we can get this way.
            self.grid_spacing_Bohr = None
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

            # Post process the text values.
            cell_volume = vol / (self.grid_dimensions[0] *
                                 self.grid_dimensions[1] *
                                 self.grid_dimensions[2] * Bohr ** 3)
            self.grid_spacing_Bohr = cell_volume ** (1 / 3)
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
                    grid3D[i, j, k, 0] = i * self.grid_spacing_Bohr
                    grid3D[i, j, k, 1] = j * self.grid_spacing_Bohr
                    grid3D[i, j, k, 2] = k * self.grid_spacing_Bohr
        return grid3D

    @staticmethod
    def get_radial_distribution_function(atoms: ase.Atoms,
                                         number_of_bins=500, rMax=None):
        """
        Calculate the radial distribution function.

        Reimplemented because the ASE implementation can't handle periodic
        boundary conditions (rMax, which is the same in all three directions,
        has to be smaller then the smallest direction in the unit cell; for
        systems with non-cubic geometry, e.g. hexagonal systems, this makes
        the RDF awfully small). Reference implementation is the ASAP3 code,
        which I would like to avoid because we then have yet another
        dependency for only one function. I do not claim this to be the
        most performant implementation of the RDF, however.

        Parameters
        ----------
        atoms
        number_of_bins
        rMax

        Returns
        -------

        """
        # Leaving this here for potential future use - this is much faster
        # because C++.
        # from asap3.analysis.rdf import RadialDistributionFunction
        # (RadialDistributionFunction(atoms, rMax, 500)).get_rdf()

        # This is quite a large RDF.
        # We may want a smaller one eventually.
        if rMax is None:
            rMax = np.min(
                    np.linalg.norm(atoms.get_cell(), axis=0)) - 0.0001

        atoms = atoms
        dr = float(rMax/number_of_bins)
        rdf = np.zeros(number_of_bins + 1)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        for i in range(0,3):
            if pbc[i]:
                if rMax > cell[i,i]:
                    raise Exception("Cannot calculate RDF with this radius. "
                                    "Please choose a smaller value.")

        # Calculate all the distances.
        # rMax/2 because this is the radius around one atom, so half the
        # distance to the next one.
        # Using neighborlists grants us access to the PBC.
        neighborlist = ase.neighborlist.NeighborList(np.zeros(len(atoms))+[rMax/2.0],
                                                     bothways=True)
        neighborlist.update(atoms)
        for i in range(0, len(atoms)):
            indices, offsets = neighborlist.get_neighbors(i)
            dm = distance.cdist([atoms.get_positions()[i]],
                                atoms.positions[indices] + offsets @ atoms.get_cell())
            index = (np.ceil(dm / dr)).astype(int)
            index = index.flatten()
            out_of_scope = index > number_of_bins
            index[out_of_scope] = 0
            for i in index:
                rdf[i] += 1

        # Normalize the RDF and calculate the distances
        rr = []
        phi = len(atoms)/ atoms.get_volume()
        norm = 4.0 * np.pi * dr * phi * len(atoms)
        for i in range(1, number_of_bins + 1):
            rr.append((i - 0.5) * dr)
            rdf[i] /= (norm * ((rr[-1] ** 2) + (dr ** 2) / 12.))

        return rdf[1:], rr
        
    @staticmethod
    def get_three_particle_correlation_function(atoms: ase.Atoms,
                                                number_of_bins,
                                                rMax=None):
        """
        Implemented as given by:
        doi.org/10.1063/5.0048450, equation 17.

        Returns
        -------

        """
        import time
        if rMax is None:
            rMax = np.min(
                    np.linalg.norm(atoms.get_cell(), axis=0)) - 0.0001

        # TPCF is a function of three radii.
        atoms = atoms
        dr = float(rMax/number_of_bins)
        tpcf = np.zeros([number_of_bins + 1, number_of_bins + 1,
                        number_of_bins + 1])
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        for i in range(0,3):
            if pbc[i]:
                if rMax > cell[i,i]:
                    raise Exception("Cannot calculate RDF with this radius. "
                                    "Please choose a smaller value.")

        # Construct a neighbor list for calculation of distances.
        # With this, the PBC are satisfied.
        neighborlist = ase.neighborlist.NeighborList(np.zeros(len(atoms))+[rMax/2.0],
                                                     bothways=True)
        neighborlist.update(atoms)

        # To calculate the TPCF we calculat the three distances between
        # three atoms. This
        start_time = time.time()
        for i in range(0, len(atoms)):

            # Debugging
            # if i == 1:
            #     break

            pos1 = atoms.get_positions()[i]
            indices, offsets = neighborlist.get_neighbors(i)
            # print(list(zip(indices, offsets)))
            neighbor_pairs = itertools.combinations(list(zip(indices, offsets)), r=2)
            neighbor_list = list(neighbor_pairs)
            pair_positions = np.array([np.concatenate((atoms.positions[pair1[0]] + \
                        pair1[1] @ atoms.get_cell(),
                       atoms.positions[pair2[0]] + \
                       pair2[1] @ atoms.get_cell()))
                              for pair1, pair2 in neighbor_list])
            pair_positions = np.reshape(pair_positions, (len(neighbor_list)*2, 3), order="C")
            all_dists = distance.cdist([pos1], pair_positions)[0]

            for idx, neighbor_pair in enumerate(neighbor_list):
                r1 = all_dists[2*idx]
                r2 = all_dists[2*idx+1]

                # We don't need to do any calculation if either of the
                # atoms are already out of range.
                if r1 < rMax and r2 < rMax:
                    r3 = distance.cdist([pair_positions[2*idx]], [pair_positions[2*idx+1]])
                    if r3 < rMax and np.abs(r1-r2) < r3 < (r1+r2):
                        # print(r1, r2, r3)
                        id1 = (np.ceil(r1 / dr)).astype(int)
                        id2 = (np.ceil(r2 / dr)).astype(int)
                        id3 = (np.ceil(r3 / dr)).astype(int)
                        tpcf[id1, id2, id3] += 1

        print("First loop", start_time-time.time())

        start_time = time.time()
        # Normalize the TPCF and calculate the distances.
        rr = np.zeros([3, number_of_bins+1, number_of_bins+1, number_of_bins+1])
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
        print("Second loop", start_time-time.time())
        return tpcf[1:, 1:, 1:], rr[:, 1:, 1:, 1:]

    @staticmethod
    def get_static_structure_factor(atoms: ase.Atoms, kgrid_dimension, kmax,
                                    radial_distribution_function=None):
        """
        Implemented according to arXiv 1606.03610v2, Eq. 6 as Fourier
        transformation of the radial distribution function.
        """
        if radial_distribution_function is None:
            radial_distribution_function = TargetBase.\
                get_radial_distribution_function(atoms)
        rdf = radial_distribution_function[0]
        radii = radial_distribution_function[1]

        structure_factor = np.zeros(kgrid_dimension)
        dk = float(kmax/kgrid_dimension)
        kpoints = (np.linspace(dk, kmax, kgrid_dimension,))


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

        grid_dimensions : list
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


