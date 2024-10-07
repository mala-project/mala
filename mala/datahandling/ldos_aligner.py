"""Align LDOS vectors to a reference."""

import os
import json

import numpy as np

from mala.common.parameters import (
    Parameters,
    DEFAULT_NP_DATA_DTYPE,
)
from mala.common.parallelizer import printout, barrier
from mala.common.physical_data import PhysicalData
from mala.datahandling.data_handler_base import DataHandlerBase
from mala.common.parallelizer import get_comm


class LDOSAligner(DataHandlerBase):
    """
    Align LDOS vectors based on when they first become non-zero.

    Optionally truncates from the left and right-side to remove redundant data.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters used to create the data handling object.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data. If None, then one will
        be created by this class.

    target_calculator : mala.targets.target.Target
        Used to do unit conversion on output data. If None, then one will
        be created by this class.
    """

    def __init__(
        self,
        parameters: Parameters,
        target_calculator=None,
        descriptor_calculator=None,
    ):
        self.ldos_parameters = parameters.targets
        super(LDOSAligner, self).__init__(
            parameters,
            target_calculator=target_calculator,
            descriptor_calculator=descriptor_calculator,
        )

    def add_snapshot(
        self,
        output_file,
        output_directory,
        snapshot_type="numpy",
    ):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        output_file : string
            File with saved numpy output array.

        output_directory : string
            Directory containing output_npy_file.

        snapshot_type : string
            Must be numpy, openPMD is not yet available for LDOS alignment.
        """
        super(LDOSAligner, self).add_snapshot(
            "",
            "",
            output_file,
            output_directory,
            add_snapshot_as="te",
            output_units="None",
            input_units="None",
            calculation_output_file="",
            snapshot_type=snapshot_type,
        )

        if snapshot_type != "numpy":
            raise Exception("Snapshot type must be numpy for LDOS alignment")

    def align_ldos_to_ref(
        self,
        save_path=None,
        save_name=None,
        save_path_ext="aligned/",
        reference_index=0,
        zero_tol=1e-5,
        left_truncate=False,
        right_truncate_value=None,
        number_of_electrons=None,
        n_shift_mse=None,
    ):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        save_path : string
            path to save the aligned LDOS vectors
        save_name : string
            naming convention for the aligned LDOS vectors
        save_path_ext : string
            additional path for the LDOS vectors (useful if
            save_path is left as default None)
        reference_index : int
            the snapshot number (in the snapshot directory list)
            to which all other LDOS vectors are aligned
        zero_tol : float
            the "zero" value for alignment / left side truncation
            always scaled by norm of reference LDOS mean
        left_truncate : bool
            whether to truncate the zero values on the LHS
        right_truncate_value : float
            right-hand energy value (based on reference LDOS vector)
            to which truncate LDOS vectors
            if None, no right-side truncation
        egrid_spacing_ev : float
            spacing of energy grid
        egrid_offset_ev : float
           original offset of energy grid
        number_of_electrons : float / int
            if not None, computes the energy shift relative to QE energies
        n_shift_mse : int
            how many energy grid points to consider when aligning LDOS
            vectors based on mean-squared error
            computed automatically if None
        """
        if self.parameters._configuration["mpi"]:
            comm = get_comm()
            rank = comm.rank
            size = comm.size
        else:
            comm = None
            rank = 0
            size = 1

        if rank == 0:
            # load in the reference snapshot
            snapshot_ref = self.parameters.snapshot_directories_list[
                reference_index
            ]
            ldos_ref = np.load(
                os.path.join(
                    snapshot_ref.output_npy_directory,
                    snapshot_ref.output_npy_file,
                ),
                mmap_mode="r",
            )

            # get the mean
            n_target = ldos_ref.shape[-1]
            ldos_ref = ldos_ref.reshape(-1, n_target)
            ldos_mean_ref = np.mean(ldos_ref, axis=0)
            zero_tol = zero_tol / np.linalg.norm(ldos_mean_ref)

            if n_shift_mse is None:
                n_shift_mse = n_target // 10

            # get the first non-zero value
            left_index_ref = np.where(ldos_mean_ref > zero_tol)[0][0]

            # get the energy grid
            emax = (
                self.ldos_parameters.ldos_gridoffset_ev
                + n_target * self.ldos_parameters.ldos_gridspacing_ev
            )
            e_grid = np.linspace(
                self.ldos_parameters.ldos_gridoffset_ev,
                emax,
                n_target,
                endpoint=False,
            )

            N_snapshots = len(self.parameters.snapshot_directories_list)

        else:
            ldos_mean_ref = None
            e_grid = None
            left_index_ref = None
            n_shift_mse = None
            N_snapshots = None
            n_target = None

        if self.parameters._configuration["mpi"]:
            # Broadcast necessary data to all processes
            ldos_mean_ref = comm.bcast(ldos_mean_ref, root=0)
            e_grid = comm.bcast(e_grid, root=0)
            left_index_ref = comm.bcast(left_index_ref, root=0)
            n_shift_mse = comm.bcast(n_shift_mse, root=0)
            N_snapshots = comm.bcast(N_snapshots, root=0)
            n_target = comm.bcast(n_target, root=0)

            local_snapshots = [i for i in range(rank, N_snapshots, size)]

        else:
            local_snapshots = range(N_snapshots)

        for idx in local_snapshots:
            snapshot = self.parameters.snapshot_directories_list[idx]
            print(f"Aligning snapshot {idx+1} of {N_snapshots}")
            ldos = np.load(
                os.path.join(
                    snapshot.output_npy_directory,
                    snapshot.output_npy_file,
                ),
                mmap_mode="r",
            )

            # get the mean
            nx = ldos.shape[0]
            ny = ldos.shape[1]
            nz = ldos.shape[2]
            ldos = ldos.reshape(-1, n_target)
            ldos_shifted = np.zeros_like(ldos)
            ldos_mean = np.mean(ldos, axis=0)

            # get the first non-zero value
            left_index = np.where(ldos_mean > zero_tol)[0][0]

            # shift the ldos
            optimal_shift = self.calc_optimal_ldos_shift(
                e_grid,
                ldos_mean,
                ldos_mean_ref,
                left_index,
                left_index_ref,
                n_shift_mse,
            )

            e_shift = optimal_shift * self.ldos_parameters.ldos_gridspacing_ev
            if optimal_shift != 0:
                ldos_shifted[:, :-optimal_shift] = ldos[:, optimal_shift:]
            else:
                ldos_shifted = ldos
            del ldos

            # truncate ldos before sudden drop
            if right_truncate_value is not None:
                e_index_cut = np.where(e_grid > right_truncate_value)[0][0]
                ldos_shifted = ldos_shifted[:, :e_index_cut]
                new_upper_egrid_lim = right_truncate_value + e_shift

            # remove zero values at start of ldos
            if left_truncate:
                # get the first non-zero value
                ldos_shifted = ldos_shifted[:, left_index_ref:]
                new_egrid_offset = (
                    self.ldos_parameters.ldos_gridoffset_ev
                    + (left_index_ref + optimal_shift)
                    * self.ldos_parameters.ldos_gridspacing_ev
                )
            else:
                new_egrid_offset = self.ldos_parameters.ldos_gridoffset_ev

            # reshape
            ldos_shifted = ldos_shifted.reshape(nx, ny, nz, -1)

            ldos_shift_info = {
                "ldos_shift_ev": round(e_shift, 4),
                "aligned_ldos_gridoffset_ev": round(new_egrid_offset, 4),
                "aligned_ldos_gridsize": np.shape(ldos_shifted)[-1],
                "aligned_ldos_gridspacing": round(
                    self.ldos_parameters.ldos_gridspacing_ev, 4
                ),
            }

            if number_of_electrons is not None:
                ldos_shift_info["energy_shift_from_qe_ev"] = round(
                    number_of_electrons * e_shift, 4
                )

            save_path = os.path.join(
                snapshot.output_npy_directory, save_path_ext
            )
            save_name = snapshot.output_npy_file

            stripped_output_file_name = snapshot.output_npy_file.replace(
                ".out", ""
            )
            ldos_shift_info_save_name = stripped_output_file_name.replace(
                ".npy", ".ldos_shift.info.json"
            )

            os.makedirs(save_path, exist_ok=True)

            if "*" in save_name:
                save_name = save_name.replace("*", str(idx))
                ldos_shift_info_save_name.replace("*", str(idx))

            target_name = os.path.join(save_path, save_name)

            self.target_calculator.write_to_numpy_file(
                target_name, ldos_shifted
            )

            with open(
                os.path.join(save_path, ldos_shift_info_save_name), "w"
            ) as f:
                json.dump(ldos_shift_info, f, indent=2)

        barrier()
    
    @staticmethod
    def calc_optimal_ldos_shift(
        e_grid,
        ldos_mean,
        ldos_mean_ref,
        left_index,
        left_index_ref,
        n_shift_mse,
    ):
        """
        Calculate the optimal amount by which to align the LDOS with reference.

        'Optimized' is currently based on minimizing the mean-square error with
        the reference, up to a cut-off (typically 10% of the full LDOS length).

        Parameters
        ----------
        e_grid : array_like
            energy grid
        ldos_mean : array_like
            mean of LDOS vector for shifting
        ldos_mean_ref : array_like
            mean of LDOS reference vector
        left_index : int
            index at which LDOS for shifting becomes non-zero
        left_index_ref : int
            index at which reference LDOS becomes non-zero
        n_shift_mse : int
            number of points to account for in MSE calculation
            for optimal LDOS shift

        Returns
        -------
        optimal_shift : int
            the optimized number of egrid points to shift the LDOS
            vector by, based on minimization of MSE with reference
        """
        shift_guess = 0
        ldos_diff = np.inf
        shift_guess = max(left_index - left_index_ref - 2, 0)
        for i in range(5):
            shift = shift_guess + i
            ldos_mean_shifted = np.zeros_like(ldos_mean)
            if shift != 0:
                ldos_mean_shifted[:-shift] = ldos_mean[shift:]
            else:
                ldos_mean_shifted = ldos_mean

            e_index_cut = max(left_index, left_index_ref) + n_shift_mse
            ldos_mean_shifted = ldos_mean_shifted[:e_index_cut]
            ldos_mean_ref = ldos_mean_ref[:e_index_cut]

            mse = np.sum((ldos_mean_shifted - ldos_mean_ref) ** 2)
            if mse < ldos_diff:
                optimal_shift = shift
                ldos_diff = mse

        return optimal_shift
