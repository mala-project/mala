"""Base class for all descriptor calculators."""
from abc import ABC, abstractmethod

import ase
import numpy as np

from mala.common.parameters import ParametersDescriptors, Parameters
from mala.common.parallelizer import printout, get_size


class Descriptor(ABC):
    """
    Base class for all descriptors available in MALA.

    Descriptors encode the atomic fingerprint of a DFT calculation.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.

    """

    def __new__(cls, params: Parameters):
        """
        Create a Descriptor instance.

        The correct type of descriptor calculator will automatically be
        instantiated by this class if possible. You can also instantiate
        the desired descriptor directly by calling upon the subclass.

        Parameters
        ----------
        params : mala.common.parametes.Parameters
            Parameters used to create this descriptor calculator.
        """
        descriptors = None
        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == Descriptor:
            if params.descriptors.descriptor_type == 'SNAP':
                from mala.descriptors.snap import SNAP
                descriptors = super(Descriptor, SNAP).__new__(SNAP)

            if descriptors is None:
                raise Exception("Unsupported descriptor calculator.")
        else:
            descriptors = super(Descriptor, cls).__new__(cls)

        return descriptors

    def __init__(self, parameters):
        self.parameters: ParametersDescriptors = parameters.descriptors
        self.fingerprint_length = -1  # so iterations will fail
        self.dbg_grid_dimensions = parameters.debug.grid_dimensions
        self.verbosity = parameters.verbosity

    @property
    def descriptors_contain_xyz(self):
        """Control whether descriptor vectors will contain xyz coordinates."""
        return self.parameters.descriptors_contain_xyz

    @descriptors_contain_xyz.setter
    def descriptors_contain_xyz(self, value):
        self.parameters.descriptors_contain_xyz = value

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert descriptors from a specified unit into the ones used in MALA.

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
        raise Exception("No unit conversion method implemented for this"
                        " descriptor type.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert descriptors from MALA units into a specified unit.

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
        raise Exception("No unit back conversion method implemented for "
                        "this descriptor type.")

    @staticmethod
    def enforce_pbc(atoms):
        """
        Explictly enforces the PBC on an ASE atoms object.

        QE (and potentially other codes?) do that internally. Meaning that the
        raw positions of atoms (in Angstrom) can lie outside of the unit cell.
        When setting up the DFT calculation, these atoms get shifted into
        the unit cell. Since we directly use these raw positions for the
        descriptor calculation, we need to enforce that in the ASE atoms
        objects, the atoms are explicitly in the unit cell.

        Parameters
        ----------
        atoms : ase.atoms
            The ASE atoms object for which the PBC need to be enforced.

        Returns
        -------
        new_atoms : ase.Atoms
            The ASE atoms object for which the PBC have been enforced.
        """
        new_atoms = atoms.copy()
        new_atoms.set_scaled_positions(new_atoms.get_scaled_positions())

        # This might be unecessary, but I think it is nice to have some sort of
        # metric here.
        rescaled_atoms = 0
        for i in range(0, len(atoms)):
            if False in (np.isclose(new_atoms[i].position,
                          atoms[i].position, atol=0.001)):
                rescaled_atoms += 1
        printout("Descriptor calculation: had to enforce periodic boundary "
                 "conditions on", rescaled_atoms, "atoms before calculation.",
                 min_verbosity=2)
        return new_atoms

    @abstractmethod
    def calculate_from_qe_out(self, qe_out_file, **kwargs):
        """
        Calculate the descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.


        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)

        """
        pass

    @abstractmethod
    def calculate_from_atoms(self, atoms, grid_dimensions):
        """
        Calculate the descriptors based on the atomic configurations.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object holding the atomic configuration.

        grid_dimensions : list
            Grid dimensions to be used, in the format [x,y,z].

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)
        """
        pass

    def get_acsd(self, descriptor_data, ldos_data):
        """
        Calculate the ACSD for given descriptors and LDOS data.

        ACSD stands for average cosine similarity distance and is a metric
        of how well the descriptors capture the local environment to a
        degree where similar vectors result in simlar LDOS vectors.

        Parameters
        ----------
        descriptor_data : numpy.ndarray
            Array containing the descriptors.

        ldos_data : numpy.ndarray
            Array containing the LDOS.

        Returns
        -------
        acsd : float
            The average cosine similarity distance.

        """
        return self._calculate_acsd(descriptor_data, ldos_data,
                                    self.parameters.acsd_points,
                                    descriptor_vectors_contain_xyz=
                                    self.descriptors_contain_xyz)

    def _setup_lammps_processors(self, nx, ny, nz):
        """
        Set up the lammps processor grid.

        Takes into account y/z-splitting.
        """
        if self.parameters._configuration["mpi"]:
            size = get_size()
            # for parallel tem need to set lammps commands: processors and
            # balance current implementation is to match lammps mpi processor
            # grid to QE processor splitting QE distributes grid points in
            # parallel as slices along z axis currently grid points fall on z
            # axix plane cutoff values in lammps this leads to some ranks
            # having 0 grid points and other having 2x gridpoints
            # balance command in lammps aleviates this issue
            # integers for plane cuts in z axis appear to be most important
            #
            # determine if nyfft flag is set so that QE also parallelizes
            # along y axis if nyfft is true lammps mpi processor grid needs to
            # be 1x{ny}x{nz} need to configure separate total_energy_module
            # with nyfft enabled
            if self.parameters.use_y_splitting > 1:
                # TODO automatically pass nyfft into QE from MALA
                # if more processors thatn y*z grid dimensions requested
                # send error. More processors than y*z grid dimensions reduces
                # efficiency and scaling of QE.
                nyfft = self.parameters.use_y_splitting
                # number of y processors is equal to nyfft
                yprocs = nyfft
                # number of z processors is equal to total processors/nyfft is
                # nyfft is used else zprocs = size
                if size % yprocs == 0:
                    zprocs = int(size/yprocs)
                else:
                    raise ValueError("Cannot evenly divide z-planes "
                                     "in y-direction")

                # check if total number of processors is greater than number of
                # grid sections produce error if number of processors is
                # greater than grid partions - will cause mismatch later in QE
                mpi_grid_sections = yprocs*zprocs
                if mpi_grid_sections < size:
                    raise ValueError("More processors than grid sections. "
                                     "This will cause a crash further in the "
                                     "calculation. Choose a total number of "
                                     "processors equal to or less than the "
                                     "total number of grid sections requsted "
                                     "for the calculation (nyfft*nz).")
                # TODO not sure what happens when size/nyfft is not integer -
                #  further testing required

                # set the mpi processor grid for lammps
                lammps_procs = f"1 {yprocs} {zprocs}"
                printout("mpi grid with nyfft: ", lammps_procs,
                         min_verbosity=2)

                # prepare y plane cuts for balance command in lammps if not
                # integer value
                if int(ny / yprocs) == (ny / yprocs):
                    ycut = 1/yprocs
                    yint = ''
                    for i in range(0, yprocs-1):
                        yvals = ((i+1)*ycut)-0.00000001
                        yint += format(yvals, ".8f")
                        yint += ' '
                else:
                    # account for remainder with uneven number of
                    # planes/processors
                    ycut = 1/yprocs
                    yrem = ny - (yprocs*int(ny/yprocs))
                    yint = ''
                    for i in range(0, yrem):
                        yvals = (((i+1)*2)*ycut)-0.00000001
                        yint += format(yvals, ".8f")
                        yint += ' '
                    for i in range(yrem, yprocs-1):
                        yvals = ((i+1+yrem)*ycut)-0.00000001
                        yint += format(yvals, ".8f")
                        yint += ' '
                # prepare z plane cuts for balance command in lammps
                if int(nz / zprocs) == (nz / zprocs):
                    zcut = 1/nz
                    zint = ''
                    for i in range(0, zprocs-1):
                        zvals = ((i + 1) * (nz / zprocs) * zcut) - 0.00000001
                        zint += format(zvals, ".8f")
                        zint += ' '
                else:
                    # account for remainder with uneven number of
                    # planes/processors
                    raise ValueError("Cannot divide z-planes on processors"
                                     " without remainder. "
                                     "This is currently unsupported.")

                    # zcut = 1/nz
                    # zrem = nz - (zprocs*int(nz/zprocs))
                    # zint = ''
                    # for i in range(0, zrem):
                    #     zvals = (((i+1)*2)*zcut)-0.00000001
                    #     zint += format(zvals, ".8f")
                    #     zint += ' '
                    # for i in range(zrem, zprocs-1):
                    #     zvals = ((i+1+zrem)*zcut)-0.00000001
                    #     zint += format(zvals, ".8f")
                    #     zint += ' '
                lammps_dict = {"lammps_procs": f"processors {lammps_procs} "
                                               f"map xyz",
                               "zbal": f"balance 1.0 y {yint} z {zint}",
                               "ngridx": nx,
                               "ngridy": ny,
                               "ngridz": nz}
            else:
                if self.parameters.use_z_splitting:
                    # when nyfft is not used only split processors along z axis
                    size = get_size()
                    zprocs = size
                    # check to make sure number of z planes is not less than
                    # processors. If more processors than planes calculation
                    # efficiency decreases
                    if nz < size:
                        raise ValueError("More processors than grid sections. "
                                         "This will cause a crash further in "
                                         "the calculation. Choose a total "
                                         "number of processors equal to or "
                                         "less than the total number of grid "
                                         "sections requsted for the "
                                         "calculation (nz).")

                    # match lammps mpi grid to be 1x1x{zprocs}
                    lammps_procs = f"1 1 {zprocs}"
                    # print("mpi grid z only: ", lammps_procs)

                    # prepare z plane cuts for balance command in lammps
                    if int(nz / zprocs) == (nz / zprocs):
                        printout("No remainder in z")
                        zcut = 1/nz
                        zint = ''
                        for i in range(0, zprocs-1):
                            zvals = ((i+1)*(nz/zprocs)*zcut)-0.00000001
                            zint += format(zvals, ".8f")
                            zint += ' '
                    else:
                        raise ValueError("Cannot divide z-planes on processors"
                                         " without remainder. "
                                         "This is currently unsupported.")
                    #     zcut = 1/nz
                    #     zrem = nz - (zprocs*int(nz/zprocs))
                    #     zint = ''
                    #     for i in range(0, zrem):
                    #         zvals = (((i+1)*2)*int(nz/zprocs)*zcut)-
                    #         0.00000001
                    #         zint += format(zvals, ".8f")
                    #         zint += ' '
                    #     for i in range(zrem, zprocs-1):
                    #         zvals = ((i+1+zrem)*zcut)-0.00000001
                    #         zint += format(zvals, ".8f")
                    #         zint += ' '
                    lammps_dict = {"lammps_procs":
                                   f"processors {lammps_procs}",
                                   "zbal": f"balance 1.0 z {zint}",
                                   "ngridx": nx,
                                   "ngridy": ny,
                                   "ngridz": nz,
                                   "switch": self.parameters.snap_switchflag}
                else:
                    lammps_dict = {"ngridx": nx,
                                   "ngridy": ny,
                                   "ngridz": nz,
                                   "switch": self.parameters.snap_switchflag}

        else:
            lammps_dict = {"ngridx": nx,
                           "ngridy": ny,
                           "ngridz": nz,
                           "switch": self.parameters.snap_switchflag}
        return lammps_dict

    @staticmethod
    def _calculate_cosine_similarities(descriptor_data, ldos_data, nr_points,
                                       descriptor_vectors_contain_xyz=True):
        """
        Calculate the raw cosine similarities for descriptor and LDOS data.

        Parameters
        ----------
        descriptor_data : numpy.ndarray
            Array containing the descriptors.

        ldos_data : numpy.ndarray
            Array containing the LDOS.

        descriptor_vectors_contain_xyz : bool
            If true, the xyz values are cut from the beginning of the
            descriptor vectors.

        nr_points : int
            The number of points for which to calculate the ACSD.
            The actual number of distances will be acsd_points x acsd_points,
            since the cosine similarity is only defined for pairs.

        Returns
        -------
        similarity_array : numpy.ndarray
            A (2,nr_points*nr_points) array containing the cosine similarities.

        """
        def calc_cosine_similarity(vector1, vector2, norm=2):
            if np.shape(vector1)[0] != np.shape(vector2)[0]:
                raise Exception("Cannot calculate similarity between vectors "
                                "of different dimenstions.")
            if np.shape(vector1)[0] == 1:
                return np.min([vector1[0], vector2[0]]) / \
                       np.max([vector1[0], vector2[0]])
            else:
                return np.dot(vector1, vector2) / \
                       (np.linalg.norm(vector1, ord=norm) *
                        np.linalg.norm(vector2, ord=norm))

        descriptor_dim = np.shape(descriptor_data)
        ldos_dim = np.shape(ldos_data)
        if len(descriptor_dim) == 4:
            descriptor_data = np.reshape(descriptor_data,
                                         (descriptor_dim[0] *
                                          descriptor_dim[1] *
                                          descriptor_dim[2],
                                          descriptor_dim[3]))
            if descriptor_vectors_contain_xyz:
                descriptor_data = descriptor_data[:, 3:]
        elif len(descriptor_dim) != 2:
            raise Exception("Cannot work with this descriptor data.")

        if len(ldos_dim) == 4:
            ldos_data = np.reshape(ldos_data, (ldos_dim[0] * ldos_dim[1] *
                                               ldos_dim[2], ldos_dim[3]))
        elif len(ldos_dim) != 2:
            raise Exception("Cannot work with this LDOS data.")

        similarity_array = []
        # Draw nr_points at random from snapshot.
        rng = np.random.default_rng()
        points_i = rng.choice(np.shape(descriptor_data)[0],
                              size=np.shape(descriptor_data)[0],
                              replace=False)
        for i in range(0, nr_points):
            # Draw another nr_points at random from snapshot.
            rng = np.random.default_rng()
            points_j = rng.choice(np.shape(descriptor_data)[0],
                                  size=np.shape(descriptor_data)[0],
                                  replace=False)

            for j in range(0, nr_points):
                # Calculate similarities between these two pairs.
                descriptor_distance = \
                    calc_cosine_similarity(
                        descriptor_data[points_i[i]],
                        descriptor_data[points_j[j]])
                ldos_distance = calc_cosine_similarity(ldos_data[points_i[i]],
                                                       ldos_data[points_j[j]])
                similarity_array.append([descriptor_distance, ldos_distance])

        return np.array(similarity_array)

    @staticmethod
    def _calculate_acsd(descriptor_data, ldos_data, acsd_points,
                        descriptor_vectors_contain_xyz=True):
        """
        Calculate the ACSD for given descriptor and LDOS data.

        ACSD stands for average cosine similarity distance and is a metric
        of how well the descriptors capture the local environment to a
        degree where similar descriptor vectors result in simlar LDOS vectors.

        Parameters
        ----------
        descriptor_data : numpy.ndarray
            Array containing the descriptors.

        ldos_data : numpy.ndarray
            Array containing the LDOS.

        descriptor_vectors_contain_xyz : bool
            If true, the xyz values are cut from the beginning of the
            descriptor vectors.

        acsd_points : int
            The number of points for which to calculate the ACSD.
            The actual number of distances will be acsd_points x acsd_points,
            since the cosine similarity is only defined for pairs.

        Returns
        -------
        acsd : float
            The average cosine similarity distance.

        """
        def distance_between_points(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        similarity_data = Descriptor.\
            _calculate_cosine_similarities(descriptor_data, ldos_data,
                                           acsd_points,
                                           descriptor_vectors_contain_xyz=
                                           descriptor_vectors_contain_xyz)
        data_size = np.shape(similarity_data)[0]
        distances = []
        for i in range(0, data_size):
            distances.append(distance_between_points(similarity_data[i, 0],
                                                     similarity_data[i, 1],
                                                     similarity_data[i, 0],
                                                     similarity_data[i, 0]))
        return np.mean(distances)


