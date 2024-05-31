"""Base class for all descriptor calculators."""

from abc import abstractmethod
from datetime import datetime
from functools import cached_property
import os

import ase
from ase.units import m
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import numpy as np
from skspatial.objects import Plane

from mala.common.parameters import ParametersDescriptors, Parameters
from mala.common.parallelizer import (
    get_comm,
    printout,
    get_rank,
    get_size,
    barrier,
    parallel_warn,
    set_lammps_instance,
)
from mala.common.physical_data import PhysicalData
from mala.descriptors.lammps_utils import set_cmdlinevars


class Descriptor(PhysicalData):
    """
    Base class for all descriptors available in MALA.

    Descriptors encode the atomic fingerprint of a DFT calculation.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.

    """

    ##############################
    # Constructors
    ##############################

    def __new__(cls, params: Parameters = None):
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
            if params.descriptors.descriptor_type == "SNAP":
                from mala.descriptors.bispectrum import Bispectrum

                parallel_warn(
                    "Using 'SNAP' as descriptors will be deprecated "
                    "starting in MALA v1.3.0. Please use 'Bispectrum' "
                    "instead.",
                    min_verbosity=0,
                    category=FutureWarning,
                )
                descriptors = super(Descriptor, Bispectrum).__new__(Bispectrum)

            if params.descriptors.descriptor_type == "Bispectrum":
                from mala.descriptors.bispectrum import Bispectrum

                descriptors = super(Descriptor, Bispectrum).__new__(Bispectrum)

            if params.descriptors.descriptor_type == "AtomicDensity":
                from mala.descriptors.atomic_density import AtomicDensity

                descriptors = super(Descriptor, AtomicDensity).__new__(
                    AtomicDensity
                )

            if params.descriptors.descriptor_type == "ACE":
                from mala.descriptors.ace import ACE

                descriptors = super(Descriptor, ACE).__new__(
                    ACE
                )

            if params.descriptors.descriptor_type == "MinterpyDescriptors":
                from mala.descriptors.minterpy_descriptors import (
                    MinterpyDescriptors,
                )

                descriptors = super(Descriptor, MinterpyDescriptors).__new__(
                    MinterpyDescriptors
                )

            if descriptors is None:
                raise Exception("Unsupported descriptor calculator.")
        else:
            descriptors = super(Descriptor, cls).__new__(cls)

        # For pickling
        setattr(descriptors, "params_arg", params)
        return descriptors

    def __getnewargs__(self):
        """
        Get the necessary arguments to call __new__.

        Used for pickling.


        Returns
        -------
        params : mala.Parameters
            The parameters object with which this object was created.
        """
        return (self.params_arg,)

    def __init__(self, parameters):
        super(Descriptor, self).__init__(parameters)
        self.parameters: ParametersDescriptors = parameters.descriptors
        self.fingerprint_length = 0  # so iterations will fail
        self.verbosity = parameters.verbosity
        self.in_format_ase = ""
        self.atoms = None
        self.voxel = None

        # If we ever have NON LAMMPS descriptors, these parameters have no
        # meaning anymore and should probably be moved to an intermediate
        # DescriptorsLAMMPS class, from which the LAMMPS descriptors inherit.
        self.lammps_temporary_input = None
        self.lammps_temporary_log = None

    ##############################
    # Properties
    ##############################

    @property
    def si_unit_conversion(self):
        """
        Numeric value of the conversion from MALA (ASE) units to SI.

        Needed for OpenPMD interface.
        """
        return m**3

    @property
    def si_dimension(self):
        """
        Dictionary containing the SI unit dimensions in OpenPMD format.

        Needed for OpenPMD interface.
        """
        import openpmd_api as io

        return {io.Unit_Dimension.L: -3}

    @property
    def descriptors_contain_xyz(self):
        """Control whether descriptor vectors will contain xyz coordinates."""
        return self.parameters.descriptors_contain_xyz

    @descriptors_contain_xyz.setter
    def descriptors_contain_xyz(self, value):
        self.parameters.descriptors_contain_xyz = value

    @cached_property
    def calculation_timestamp(self):
        """
        Timestamp of calculation start.

        Used to distinguish multiple LAMMPS runs performed in the same
        directory. Since the interface is file based, this timestamp prevents
        problems with slightly
        """
        if get_rank() == 0:
            timestamp = datetime.timestamp(datetime.utcnow())
        else:
            timestamp = None

        if self.parameters._configuration["mpi"]:
            timestamp = get_comm().bcast(timestamp, root=0)
        return datetime.fromtimestamp(timestamp).strftime("%F-%H-%M-%S-%f")[
            :-3
        ]

    ##############################
    # Methods
    ##############################

    # File I/O
    ##########

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
        raise Exception(
            "No unit conversion method implemented for this"
            " descriptor type."
        )

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
        raise Exception(
            "No unit back conversion method implemented for "
            "this descriptor type."
        )

    # Calculations
    ##############

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
            if False in (
                np.isclose(
                    new_atoms[i].position, atoms[i].position, atol=0.001
                )
            ):
                rescaled_atoms += 1
        printout(
            "Descriptor calculation: had to enforce periodic boundary "
            "conditions on",
            rescaled_atoms,
            "atoms before calculation.",
            min_verbosity=2,
        )
        return new_atoms

    def calculate_from_qe_out(
        self, qe_out_file, working_directory=".", **kwargs
    ):
        """
        Calculate the descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        working_directory : string
            A directory in which to write the output of the LAMMPS calculation.
            Usually the local directory should suffice, given that there
            are no multiple instances running in the same directory.

        kwargs : dict
            A collection of keyword arguments, that are mainly used for
            debugging and development. Different types of descriptors
            may support different keyword arguments. Commonly supported
            are

            - "use_fp64": To use enforce floating point 64 precision for
              descriptors.
            - "keep_logs": To not delete temporary files created during
              LAMMPS calculation of descriptors.

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)

        """
        self.in_format_ase = "espresso-out"
        printout("Calculating descriptors from", qe_out_file, min_verbosity=0)
        # We get the atomic information by using ASE.
        self.atoms = ase.io.read(qe_out_file, format=self.in_format_ase)

        # Enforcing / Checking PBC on the read atoms.
        self.atoms = self.enforce_pbc(self.atoms)

        # Get the grid dimensions.
        if "grid_dimensions" in kwargs.keys():
            self.grid_dimensions = kwargs["grid_dimensions"]

            # Deleting this keyword from the list to avoid conflict with
            # dict below.
            del kwargs["grid_dimensions"]
        else:
            qe_outfile = open(qe_out_file, "r")
            lines = qe_outfile.readlines()
            self.grid_dimensions = [0, 0, 0]

            for line in lines:
                if "FFT dimensions" in line:
                    tmp = line.split("(")[1].split(")")[0]
                    self.grid_dimensions[0] = int(tmp.split(",")[0])
                    self.grid_dimensions[1] = int(tmp.split(",")[1])
                    self.grid_dimensions[2] = int(tmp.split(",")[2])
                    break

        self.voxel = self.atoms.cell.copy()
        self.voxel[0] = self.voxel[0] / (self.grid_dimensions[0])
        self.voxel[1] = self.voxel[1] / (self.grid_dimensions[1])
        self.voxel[2] = self.voxel[2] / (self.grid_dimensions[2])

        return self._calculate(working_directory, **kwargs)

    def calculate_from_atoms(
        self, atoms, grid_dimensions, working_directory=".", **kwargs
    ):
        """
        Calculate the bispectrum descriptors based on atomic configurations.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object holding the atomic configuration.

        grid_dimensions : list
            Grid dimensions to be used, in the format [x,y,z].

        working_directory : string
            A directory in which to write the output of the LAMMPS calculation.
            Usually the local directory should suffice, given that there
            are no multiple instances running in the same directory.

        kwargs : dict
            A collection of keyword arguments, that are mainly used for
            debugging and development. Different types of descriptors
            may support different keyword arguments. Commonly supported
            are

            - "use_fp64": To use enforce floating point 64 precision for
              descriptors.
            - "keep_logs": To not delete temporary files created during
              LAMMPS calculation of descriptors.

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)
        """
        # Enforcing / Checking PBC on the input atoms.
        self.atoms = self.enforce_pbc(atoms)
        self.grid_dimensions = grid_dimensions
        self.voxel = self.atoms.cell.copy()
        self.voxel[0] = self.voxel[0] / (self.grid_dimensions[0])
        self.voxel[1] = self.voxel[1] / (self.grid_dimensions[1])
        self.voxel[2] = self.voxel[2] / (self.grid_dimensions[2])
        return self._calculate(working_directory, **kwargs)

    def gather_descriptors(self, descriptors_np, use_pickled_comm=False):
        """
        Gathers all descriptors on rank 0 and sorts them.

        This is useful for e.g. parallel preprocessing.
        This function removes the extra 3 components that come from parallel
        processing.
        I.e. if we have 91 bispectrum descriptors, LAMMPS directly outputs us
        97 (in parallel mode), and this function returns 94, as to retain the
        3 x,y,z ones we by default include.

        Parameters
        ----------
        descriptors_np : numpy.array
            Numpy array with the descriptors of this ranks local grid.

        use_pickled_comm : bool
            If True, the pickled communication route from mpi4py is used.
            If False, a Recv/Sendv combination is used. I am not entirely
            sure what is faster. Technically Recv/Sendv should be faster,
            but I doubt my implementation is all that optimal. For the pickled
            route we can use gather(), which should be fairly quick.
            However, for large grids, one CANNOT use the pickled route;
            too large python objects will break it. Therefore, I am setting
            the Recv/Sendv route as default.
        """
        # Barrier to make sure all ranks have descriptors..
        comm = get_comm()
        barrier()

        # Gather the descriptors into a list.
        if use_pickled_comm:
            all_descriptors_list = comm.gather(descriptors_np, root=0)
        else:
            sendcounts = np.array(
                comm.gather(np.shape(descriptors_np)[0], root=0)
            )
            raw_feature_length = self.fingerprint_length + 3

            if get_rank() == 0:
                # print("sendcounts: {}, total: {}".format(sendcounts,
                #                                          sum(sendcounts)))

                # Preparing the list of buffers.
                all_descriptors_list = []
                for i in range(0, get_size()):
                    all_descriptors_list.append(
                        np.empty(
                            sendcounts[i] * raw_feature_length,
                            dtype=descriptors_np.dtype,
                        )
                    )

                # No MPI necessary for first rank. For all the others,
                # collect the buffers.
                all_descriptors_list[0] = descriptors_np
                for i in range(1, get_size()):
                    comm.Recv(all_descriptors_list[i], source=i, tag=100 + i)
                    all_descriptors_list[i] = np.reshape(
                        all_descriptors_list[i],
                        (sendcounts[i], raw_feature_length),
                    )
            else:
                comm.Send(descriptors_np, dest=0, tag=get_rank() + 100)
            barrier()

        # if get_rank() == 0:
        #     printout(np.shape(all_descriptors_list[0]))
        #     printout(np.shape(all_descriptors_list[1]))
        #     printout(np.shape(all_descriptors_list[2]))
        #     printout(np.shape(all_descriptors_list[3]))

        # Dummy for the other ranks.
        # (For now, might later simply broadcast to other ranks).
        descriptors_full = np.zeros([1, 1, 1, 1])

        # Reorder the list.
        if get_rank() == 0:
            # Prepare the descriptor array.
            nx = self.grid_dimensions[0]
            ny = self.grid_dimensions[1]
            nz = self.grid_dimensions[2]
            descriptors_full = np.zeros([nx, ny, nz, self.fingerprint_length])
            # Fill the full SNAP descriptors array.
            for idx, local_grid in enumerate(all_descriptors_list):
                # We glue the individual cells back together, and transpose.
                first_x = int(local_grid[0][0])
                first_y = int(local_grid[0][1])
                first_z = int(local_grid[0][2])
                last_x = int(local_grid[-1][0]) + 1
                last_y = int(local_grid[-1][1]) + 1
                last_z = int(local_grid[-1][2]) + 1
                descriptors_full[
                    first_x:last_x, first_y:last_y, first_z:last_z
                ] = np.reshape(
                    local_grid[:, 3:],
                    [
                        last_z - first_z,
                        last_y - first_y,
                        last_x - first_x,
                        self.fingerprint_length,
                    ],
                ).transpose(
                    [2, 1, 0, 3]
                )

                # Leaving this in here for debugging purposes.
                # This is the slow way to reshape the descriptors.
                # for entry in local_grid:
                #     x = int(entry[0])
                #     y = int(entry[1])
                #     z = int(entry[2])
                #     descriptors_full[x, y, z] = entry[3:]
        if self.parameters.descriptors_contain_xyz:
            return descriptors_full
        else:
            return descriptors_full[:, :, :, 3:]

    def convert_local_to_3d(self, descriptors_np):
        """
        Convert the desciptors as done in the gather function, but per rank.

        This is useful for e.g. parallel preprocessing.
        This function removes the extra 3 components that come from parallel
        processing.
        I.e. if we have 91 bispectrum descriptors, LAMMPS directly outputs us
        97 (in parallel mode), and this function returns 94, as to retain the
        3 x,y,z ones we by default include.

        Parameters
        ----------
        descriptors_np : numpy.array
            Numpy array with the descriptors of this ranks local grid.
        """
        local_offset = [None, None, None]
        local_reach = [None, None, None]
        local_offset[0] = int(descriptors_np[0][0])
        local_offset[1] = int(descriptors_np[0][1])
        local_offset[2] = int(descriptors_np[0][2])
        local_reach[0] = int(descriptors_np[-1][0]) + 1
        local_reach[1] = int(descriptors_np[-1][1]) + 1
        local_reach[2] = int(descriptors_np[-1][2]) + 1
        nx = local_reach[0] - local_offset[0]
        ny = local_reach[1] - local_offset[1]
        nz = local_reach[2] - local_offset[2]

        descriptors_full = np.zeros([nx, ny, nz, self.fingerprint_length])

        descriptors_full[0:nx, 0:ny, 0:nz] = np.reshape(
            descriptors_np[:, 3:], [nz, ny, nx, self.fingerprint_length]
        ).transpose([2, 1, 0, 3])
        return descriptors_full, local_offset, local_reach

    # Private methods
    #################

    def _process_loaded_array(self, array, units=None):
        array *= self.convert_units(1, in_units=units)

    def _process_loaded_dimensions(self, array_dimensions):
        if self.descriptors_contain_xyz:
            return (
                array_dimensions[0],
                array_dimensions[1],
                array_dimensions[2],
                array_dimensions[3] - 3,
            )
        else:
            return array_dimensions

    def _set_geometry_info(self, mesh):
        # Geometry: Save the cell parameters and angles of the grid.
        if self.atoms is not None:
            import openpmd_api as io

            self.voxel = self.atoms.cell.copy()
            self.voxel[0] = self.voxel[0] / (self.grid_dimensions[0])
            self.voxel[1] = self.voxel[1] / (self.grid_dimensions[1])
            self.voxel[2] = self.voxel[2] / (self.grid_dimensions[2])

            mesh.geometry = io.Geometry.cartesian
            mesh.grid_spacing = self.voxel.cellpar()[0:3]
            mesh.set_attribute("angles", self.voxel.cellpar()[3:])

    def _get_atoms(self):
        return self.atoms

    def _feature_mask(self):
        if self.descriptors_contain_xyz:
            return 3
        else:
            return 0

    def _setup_lammps(self, nx, ny, nz, lammps_dict):
        """
        Set up the lammps processor grid.

        Takes into account y/z-splitting.
        """
        from lammps import lammps

        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = [
            "-screen",
            "none",
            "-log",
            self.lammps_temporary_log,
        ]
        lammps_dict["atom_config_fname"] = self.lammps_temporary_input

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
                    zprocs = int(size / yprocs)
                else:
                    raise ValueError(
                        "Cannot evenly divide z-planes in y-direction"
                    )

                # check if total number of processors is greater than number of
                # grid sections produce error if number of processors is
                # greater than grid partions - will cause mismatch later in QE
                mpi_grid_sections = yprocs * zprocs
                if mpi_grid_sections < size:
                    raise ValueError(
                        "More processors than grid sections. "
                        "This will cause a crash further in the "
                        "calculation. Choose a total number of "
                        "processors equal to or less than the "
                        "total number of grid sections requsted "
                        "for the calculation (nyfft*nz)."
                    )
                # TODO not sure what happens when size/nyfft is not integer -
                #  further testing required

                # set the mpi processor grid for lammps
                lammps_procs = f"1 {yprocs} {zprocs}"
                printout(
                    "mpi grid with nyfft: ", lammps_procs, min_verbosity=2
                )

                # prepare y plane cuts for balance command in lammps if not
                # integer value
                if int(ny / yprocs) == (ny / yprocs):
                    ycut = 1 / yprocs
                    yint = ""
                    for i in range(0, yprocs - 1):
                        yvals = ((i + 1) * ycut) - 0.00000001
                        yint += format(yvals, ".8f")
                        yint += " "
                else:
                    # account for remainder with uneven number of
                    # planes/processors
                    ycut = 1 / yprocs
                    yrem = ny - (yprocs * int(ny / yprocs))
                    yint = ""
                    for i in range(0, yrem):
                        yvals = (((i + 1) * 2) * ycut) - 0.00000001
                        yint += format(yvals, ".8f")
                        yint += " "
                    for i in range(yrem, yprocs - 1):
                        yvals = ((i + 1 + yrem) * ycut) - 0.00000001
                        yint += format(yvals, ".8f")
                        yint += " "
                # prepare z plane cuts for balance command in lammps
                if int(nz / zprocs) == (nz / zprocs):
                    zcut = 1 / nz
                    zint = ""
                    for i in range(0, zprocs - 1):
                        zvals = ((i + 1) * (nz / zprocs) * zcut) - 0.00000001
                        zint += format(zvals, ".8f")
                        zint += " "
                else:
                    # account for remainder with uneven number of
                    # planes/processors
                    raise ValueError(
                        "Cannot divide z-planes on processors"
                        " without remainder. "
                        "This is currently unsupported."
                    )

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
                lammps_dict["lammps_procs"] = (
                    f"processors {lammps_procs} " f"map xyz"
                )
                lammps_dict["zbal"] = f"balance 1.0 y {yint} z {zint}"
                lammps_dict["ngridx"] = nx
                lammps_dict["ngridy"] = ny
                lammps_dict["ngridz"] = nz
                lammps_dict["switch"] = self.parameters.bispectrum_switchflag

            else:
                if self.parameters.use_z_splitting:
                    # when nyfft is not used only split processors along z axis
                    size = get_size()
                    zprocs = size
                    # check to make sure number of z planes is not less than
                    # processors. If more processors than planes calculation
                    # efficiency decreases
                    if nz < size:
                        raise ValueError(
                            "More processors than grid sections. "
                            "This will cause a crash further in "
                            "the calculation. Choose a total "
                            "number of processors equal to or "
                            "less than the total number of grid "
                            "sections requsted for the "
                            "calculation (nz)."
                        )

                    # match lammps mpi grid to be 1x1x{zprocs}
                    lammps_procs = f"1 1 {zprocs}"
                    # print("mpi grid z only: ", lammps_procs)

                    # prepare z plane cuts for balance command in lammps
                    if int(nz / zprocs) == (nz / zprocs):
                        printout("No remainder in z")
                        zcut = 1 / nz
                        zint = ""
                        for i in range(0, zprocs - 1):
                            zvals = (
                                (i + 1) * (nz / zprocs) * zcut
                            ) - 0.00000001
                            zint += format(zvals, ".8f")
                            zint += " "
                    else:
                        # raise ValueError("Cannot divide z-planes on processors"
                        #                 " without remainder. "
                        #                 "This is currently unsupported.")
                        zcut = 1 / nz
                        zrem = nz - (zprocs * int(nz / zprocs))
                        zint = ""
                        for i in range(0, zrem):
                            zvals = (
                                ((i + 1) * (int(nz / zprocs) + 1)) * zcut
                            ) - 0.00000001
                            zint += format(zvals, ".8f")
                            zint += " "
                        for i in range(zrem, zprocs - 1):
                            zvals = (
                                ((i + 1) * int(nz / zprocs) + zrem) * zcut
                            ) - 0.00000001
                            zint += format(zvals, ".8f")
                            zint += " "

                    lammps_dict["lammps_procs"] = f"processors {lammps_procs}"
                    lammps_dict["zbal"] = f"balance 1.0 z {zint}"
                    lammps_dict["ngridx"] = nx
                    lammps_dict["ngridy"] = ny
                    lammps_dict["ngridz"] = nz
                    lammps_dict["switch"] = (
                        self.parameters.bispectrum_switchflag
                    )

                else:
                    lammps_dict["ngridx"] = nx
                    lammps_dict["ngridy"] = ny
                    lammps_dict["ngridz"] = nz
                    lammps_dict["switch"] = (
                        self.parameters.bispectrum_switchflag
                    )

        else:
            size = 1
            lammps_dict["ngridx"] = nx
            lammps_dict["ngridy"] = ny
            lammps_dict["ngridz"] = nz
            lammps_dict["switch"] = self.parameters.bispectrum_switchflag
        if self.parameters._configuration["gpu"]:
            # Tell Kokkos to use one GPU.
            lmp_cmdargs.append("-k")
            lmp_cmdargs.append("on")
            lmp_cmdargs.append("g")
            lmp_cmdargs.append(str(size))

            # Tell LAMMPS to use Kokkos versions of those commands for
            # which a Kokkos version exists.
            lmp_cmdargs.append("-sf")
            lmp_cmdargs.append("kk")
            pass

        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs, lammps_dict)

        lmp = lammps(cmdargs=lmp_cmdargs)
        set_lammps_instance(lmp)

        return lmp

    def _clean_calculation(self, lmp, keep_logs):
        lmp.close()
        if not keep_logs:
            if get_rank() == 0:
                os.remove(self.lammps_temporary_log)
                os.remove(self.lammps_temporary_input)

        # Reset timestamp for potential next calculation using same LAMMPS
        # object.
        del self.calculation_timestamp

    def _setup_atom_list(self):
        """
        Set up a list of atoms potentially relevant for descriptor calculation.

        If periodic boundary conditions are used, which is usually the case
        for MALA simulation, one has to compute descriptors by also
        incorporating atoms from neighboring cells.

        FURTHER OPTIMIZATION: Probably not that much, this mostly already uses
        optimized python functions.
        """
        if np.any(self.atoms.pbc):

            # To determine the list of relevant atoms we first take the edges
            # of the simulation cell and use them to determine all cells
            # which hold atoms that _may_ be relevant for the calculation.
            edges = list(
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0],
                    ]
                )
                * np.array(self.grid_dimensions)
            )
            all_cells_list = None

            # For each edge point create a neighborhoodlist to all cells
            # given by the cutoff radius.
            for edge in edges:
                edge_point = self._grid_to_coord(edge)
                neighborlist = NeighborList(
                    np.zeros(len(self.atoms) + 1)
                    + [self.parameters.atomic_density_cutoff],
                    bothways=True,
                    self_interaction=False,
                    primitive=NewPrimitiveNeighborList,
                )

                atoms_with_grid_point = self.atoms.copy()

                # Construct a ghost atom representing the grid point.
                atoms_with_grid_point.append(ase.Atom("H", edge_point))
                neighborlist.update(atoms_with_grid_point)
                indices, offsets = neighborlist.get_neighbors(len(self.atoms))

                # Incrementally fill the list containing all cells to be
                # considered.
                if all_cells_list is None:
                    all_cells_list = np.unique(offsets, axis=0)
                else:
                    all_cells_list = np.concatenate(
                        (all_cells_list, np.unique(offsets, axis=0))
                    )

            # Delete the original cell from the list of all cells.
            # This is to avoid double checking of atoms below.
            all_cells = np.unique(all_cells_list, axis=0)
            idx = 0
            for a in range(0, len(all_cells)):
                if (all_cells[a, :] == np.array([0, 0, 0])).all():
                    break
                idx += 1
            all_cells = np.delete(all_cells, idx, axis=0)

            # Create an object to hold all relevant atoms.
            # First, instantiate it by filling it will all atoms from all
            # potentiall relevant cells, as identified above.
            all_atoms = None
            for a in range(0, len(self.atoms)):
                if all_atoms is None:
                    all_atoms = (
                        self.atoms.positions[a]
                        + all_cells @ self.atoms.get_cell()
                    )
                else:
                    all_atoms = np.concatenate(
                        (
                            all_atoms,
                            self.atoms.positions[a]
                            + all_cells @ self.atoms.get_cell(),
                        )
                    )

            # Next, construct the planes forming the unit cell.
            # Atoms from neighboring cells are only included in the list of
            # all relevant atoms, if they have a distance to any of these
            # planes smaller than the cutoff radius. Elsewise, they would
            # not be included in the eventual calculation anyhow.
            planes = [
                [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                [
                    [self.grid_dimensions[0], 1, 0],
                    [self.grid_dimensions[0], 0, 1],
                    self.grid_dimensions,
                ],
                [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
                [
                    [1, self.grid_dimensions[1], 0],
                    [0, self.grid_dimensions[1], 1],
                    self.grid_dimensions,
                ],
                [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                [
                    [1, 0, self.grid_dimensions[2]],
                    [0, 1, self.grid_dimensions[2]],
                    self.grid_dimensions,
                ],
            ]
            all_distances = []
            for plane in planes:
                curplane = Plane.from_points(
                    self._grid_to_coord(plane[0]),
                    self._grid_to_coord(plane[1]),
                    self._grid_to_coord(plane[2]),
                )
                distances = []

                # TODO: This may be optimized, and formulated in an array
                # operation.
                for a in range(np.shape(all_atoms)[0]):
                    distances.append(curplane.distance_point(all_atoms[a]))
                all_distances.append(distances)
            all_distances = np.array(all_distances)
            all_distances = np.min(all_distances, axis=0)
            all_atoms = np.squeeze(
                all_atoms[
                    np.argwhere(
                        all_distances < self.parameters.atomic_density_cutoff
                    ),
                    :,
                ]
            )
            return np.concatenate((all_atoms, self.atoms.positions))

        else:
            # If no PBC are used, only consider a single cell.
            return self.atoms.positions

    def _grid_to_coord(self, gridpoint):
        # Convert grid indices to real space grid point.
        i = gridpoint[0]
        j = gridpoint[1]
        k = gridpoint[2]
        # Orthorhombic cells and triclinic ones have
        # to be treated differently, see domain.cpp

        if self.atoms.cell.orthorhombic:
            return np.diag(self.voxel) * [i, j, k]
        else:
            ret = [0, 0, 0]
            ret[0] = (
                i / self.grid_dimensions[0] * self.atoms.cell[0, 0]
                + j / self.grid_dimensions[1] * self.atoms.cell[1, 0]
                + k / self.grid_dimensions[2] * self.atoms.cell[2, 0]
            )
            ret[1] = (
                j / self.grid_dimensions[1] * self.atoms.cell[1, 1]
                + k / self.grid_dimensions[2] * self.atoms.cell[1, 2]
            )
            ret[2] = k / self.grid_dimensions[2] * self.atoms.cell[2, 2]
            return np.array(ret)

    @abstractmethod
    def _calculate(self, outdir, **kwargs):
        pass

    def _set_feature_size_from_array(self, array):
        self.fingerprint_length = np.shape(array)[-1]
