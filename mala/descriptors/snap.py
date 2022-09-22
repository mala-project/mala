"""SNAP descriptor class."""
import os

import ase
import ase.io
try:
    from lammps import lammps
    # For version compatibility; older lammps versions (the serial version
    # we still use on some machines) do not have these constants.
    try:
        from lammps import constants as lammps_constants
    except ImportError:
        pass
except ModuleNotFoundError:
    pass
import numpy as np

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor
from mala.common.parallelizer import get_comm, printout, get_rank, get_size, \
    barrier


class SNAP(Descriptor):
    """Class for calculation and parsing of SNAP descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(SNAP, self).__init__(parameters)
        self.in_format_ase = ""
        self.grid_dimensions = []

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a SNAP descriptor.

        Since these do not really have units this function does nothing yet.

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
        if in_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for SNAP.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a SNAP descriptor.

        Since these do not really have units this function does nothing yet.

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
        if out_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for SNAP.")

    def calculate_from_qe_out(self, qe_out_file, working_directory=".",
                              **kwargs):
        """
        Calculate the SNAP descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        working_directory : string
            A directory in which to write the output of the LAMMPS calculation.
            Usually the local directory should suffice, given that there
            are no multiple instances running in the same directory.

        working_directory : string
            A directory in which to perform the LAMMPS calculation.
            Optional, if None, the QE out directory will be used.

        Returns
        -------
        snap_descriptors : numpy.array
            Numpy array containing the SNAP descriptors with the dimension
            (x,y,z,snap_dimension)

        """
        self.in_format_ase = "espresso-out"
        printout("Calculating SNAP descriptors from", qe_out_file,
                 min_verbosity=0)
        # We get the atomic information by using ASE.
        atoms = ase.io.read(qe_out_file, format=self.in_format_ase)

        # Enforcing / Checking PBC on the read atoms.
        atoms = self.enforce_pbc(atoms)

        # Get the grid dimensions.
        qe_outfile = open(qe_out_file, "r")
        lines = qe_outfile.readlines()
        nx = 0
        ny = 0
        nz = 0

        for line in lines:
            if "FFT dimensions" in line:
                tmp = line.split("(")[1].split(")")[0]
                nx = int(tmp.split(",")[0])
                ny = int(tmp.split(",")[1])
                nz = int(tmp.split(",")[2])
                break

        return self.__calculate_snap(atoms,
                                     working_directory, [nx, ny, nz])

    def calculate_from_atoms(self, atoms, grid_dimensions,
                             working_directory="."):
        """
        Calculate the SNAP descriptors based on the atomic configurations.

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

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)
        """
        # Enforcing / Checking PBC on the input atoms.
        atoms = self.enforce_pbc(atoms)
        return self.__calculate_snap(atoms, working_directory,
                                     grid_dimensions)

    def gather_descriptors(self, snap_descriptors_np, use_pickled_comm=False):
        """
        Gathers all SNAP descriptors on rank 0 and sorts them.

        This is useful for e.g. parallel preprocessing.
        This function removes the extra 3 components that come from parallel
        processing.
        I.e. if we have 91 SNAP descriptors, LAMMPS directly outputs us
        97 (in parallel mode), and this function returns 94, as to retain the
        3 x,y,z ones we by default include.

        Parameters
        ----------
        snap_descriptors_np : numpy.array
            Numpy array with the SNAP descriptors of this ranks local grid.

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

        # Gather the SNAP descriptors into a list.
        if use_pickled_comm:
            all_snap_descriptors_list = comm.gather(snap_descriptors_np,
                                                    root=0)
        else:
            sendcounts = np.array(comm.gather(np.shape(snap_descriptors_np)[0],
                                              root=0))
            raw_feature_length = self.fingerprint_length+3

            if get_rank() == 0:
                # print("sendcounts: {}, total: {}".format(sendcounts,
                #                                          sum(sendcounts)))

                # Preparing the list of buffers.
                all_snap_descriptors_list = []
                for i in range(0, get_size()):
                    all_snap_descriptors_list.append(
                        np.empty(sendcounts[i] * raw_feature_length,
                                 dtype=np.float64))

                # No MPI necessary for first rank. For all the others,
                # collect the buffers.
                all_snap_descriptors_list[0] = snap_descriptors_np
                for i in range(1, get_size()):
                    comm.Recv(all_snap_descriptors_list[i], source=i,
                              tag=100+i)
                    all_snap_descriptors_list[i] = \
                        np.reshape(all_snap_descriptors_list[i],
                                   (sendcounts[i], raw_feature_length))
            else:
                comm.Send(snap_descriptors_np, dest=0, tag=get_rank()+100)
            barrier()

        # if get_rank() == 0:
        #     printout(np.shape(all_snap_descriptors_list[0]))
        #     printout(np.shape(all_snap_descriptors_list[1]))
        #     printout(np.shape(all_snap_descriptors_list[2]))
        #     printout(np.shape(all_snap_descriptors_list[3]))

        # Dummy for the other ranks.
        # (For now, might later simply broadcast to other ranks).
        snap_descriptors_full = np.zeros([1, 1, 1, 1])

        # Reorder the list.
        if get_rank() == 0:
            # Prepare the SNAP descriptor array.
            nx = self.grid_dimensions[0]
            ny = self.grid_dimensions[1]
            nz = self.grid_dimensions[2]
            snap_descriptors_full = np.zeros(
                [nx, ny, nz, self.fingerprint_length])
            # Fill the full SNAP descriptors array.
            for idx, local_snap_grid in enumerate(all_snap_descriptors_list):
                # We glue the individual cells back together, and transpose.
                first_x = int(local_snap_grid[0][0])
                first_y = int(local_snap_grid[0][1])
                first_z = int(local_snap_grid[0][2])
                last_x = int(local_snap_grid[-1][0])+1
                last_y = int(local_snap_grid[-1][1])+1
                last_z = int(local_snap_grid[-1][2])+1
                snap_descriptors_full[first_x:last_x,
                                      first_y:last_y,
                                      first_z:last_z] = \
                    np.reshape(local_snap_grid[:, 3:],
                               [last_z-first_z, last_y-first_y, last_x-first_x,
                                self.fingerprint_length]).\
                    transpose([2, 1, 0, 3])

                # Leaving this in here for debugging purposes.
                # This is the slow way to reshape the descriptors.
                # for entry in local_snap_grid:
                #     x = int(entry[0])
                #     y = int(entry[1])
                #     z = int(entry[2])
                #     snap_descriptors_full[x, y, z] = entry[3:]
        if self.parameters.descriptors_contain_xyz:
            return snap_descriptors_full
        else:
            return snap_descriptors_full[:, :, :, 3:]

    def __calculate_snap(self, atoms, outdir, grid_dimensions):
        """Perform actual SNAP calculation."""
        from lammps import lammps
        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        # We also need to know how big the grid is.
        # Iterating directly through the file is slow, but the
        # grid information is at the top (around line 200).
        if len(self.dbg_grid_dimensions) == 3:
            nx = self.dbg_grid_dimensions[0]
            ny = self.dbg_grid_dimensions[1]
            nz = self.dbg_grid_dimensions[2]
        else:
            nx = grid_dimensions[0]
            ny = grid_dimensions[1]
            nz = grid_dimensions[2]

        # Build LAMMPS arguments from the data we read.
        lmp_cmdargs = ["-screen", "none", "-log",
                       os.path.join(outdir, "lammps_bgrid_log.tmp")]

        # LAMMPS processor grid filled by parent class.
        lammps_dict = self._setup_lammps_processors(nx, ny, nz)

        # Set the values not already filled in the LAMMPS setup.
        lammps_dict["twojmax"] = self.parameters.twojmax
        lammps_dict["rcutfac"] = self.parameters.rcutfac
        lammps_dict["atom_config_fname"] = ase_out_path

        lmp_cmdargs = set_cmdlinevars(lmp_cmdargs, lammps_dict)

        # Build the LAMMPS object.
        lmp = lammps(cmdargs=lmp_cmdargs)

        # An empty string means that the user wants to use the standard input.
        # What that is differs depending on serial/parallel execution.
        if self.parameters.lammps_compute_file == "":
            filepath = __file__.split("snap")[0]
            if self.parameters._configuration["mpi"]:
                if self.parameters.use_z_splitting:
                    self.parameters.lammps_compute_file = \
                        os.path.join(filepath, "in.bgridlocal.python")
                else:
                    self.parameters.lammps_compute_file = \
                        os.path.join(filepath,
                                     "in.bgridlocal_defaultproc.python")
            else:
                self.parameters.lammps_compute_file = \
                    os.path.join(filepath, "in.bgrid.python")

        # Do the LAMMPS calculation.
        lmp.file(self.parameters.lammps_compute_file)

        # Set things not accessible from LAMMPS
        # First 3 cols are x, y, z, coords
        ncols0 = 3

        # Analytical relation for fingerprint length
        ncoeff = (self.parameters.twojmax+2) * \
                 (self.parameters.twojmax+3)*(self.parameters.twojmax+4)
        ncoeff = ncoeff // 24   # integer division
        self.fingerprint_length = ncols0+ncoeff

        # Extract data from LAMMPS calculation.
        # This is different for the parallel and the serial case.
        # In the serial case we can expect to have a full SNAP array at the
        # end of this function.
        # This is not necessarily true for the parallel case.
        if self.parameters._configuration["mpi"]:
            nrows_local = extract_compute_np(lmp, "bgridlocal",
                                             lammps_constants.LMP_STYLE_LOCAL,
                                             lammps_constants.LMP_SIZE_ROWS)
            ncols_local = extract_compute_np(lmp, "bgridlocal",
                                             lammps_constants.LMP_STYLE_LOCAL,
                                             lammps_constants.LMP_SIZE_COLS)
            if ncols_local != self.fingerprint_length + 3:
                raise Exception("Inconsistent number of features.")

            snap_descriptors_np = \
                extract_compute_np(lmp, "bgridlocal",
                                   lammps_constants.LMP_STYLE_LOCAL, 2,
                                   array_shape=(nrows_local, ncols_local))

            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]

            # If I directly return the descriptors, this sometimes leads
            # to errors, because presumably the python garbage collection
            # deallocates memory too quickly. This copy is more memory
            # hungry, and we might have to tackle this later on, but
            # for now it works.
            return snap_descriptors_np.copy(), nrows_local

        else:
            # Extract data from LAMMPS calculation.
            snap_descriptors_np = \
                extract_compute_np(lmp, "bgrid", 0, 2,
                                   (nz, ny, nx, self.fingerprint_length))
            # switch from x-fastest to z-fastest order (swaps 0th and 2nd
            # dimension)
            snap_descriptors_np = snap_descriptors_np.transpose([2, 1, 0, 3])
            # Copy the grid dimensions only at the end.
            self.grid_dimensions = [nx, ny, nz]
            # If I directly return the descriptors, this sometimes leads
            # to errors, because presumably the python garbage collection
            # deallocates memory too quickly. This copy is more memory
            # hungry, and we might have to tackle this later on, but
            # for now it works.
            # I thought the transpose would take care of that, but apparently
            # it does not necessarily do that - so we have do go down
            # that route.
            if self.parameters.descriptors_contain_xyz:
                return snap_descriptors_np.copy(), nx*ny*nz
            else:
                return snap_descriptors_np[:, :, :, 3:].copy(), nx*ny*nz
