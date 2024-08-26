"Aligns LDOS vectors" ""

import os
import json

import numpy as np

from mala.common.parameters import (
    Parameters,
    DEFAULT_NP_DATA_DTYPE,
)
from mala.common.parallelizer import printout
from mala.common.physical_data import PhysicalData
from mala.datahandling.data_handler_base import DataHandlerBase
from mala.common.parallelizer import get_comm


class LDOSAlign(DataHandlerBase):
    """
    Mixes data between snapshots for improved lazy-loading training.

    This is a DISK operation - new, shuffled snapshots will be created on disk.

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
        super(LDOSAlign, self).__init__(
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
            Either "numpy" or "openpmd" based on what kind of files you
            want to operate on.
        """
        super(LDOSAlign, self).add_snapshot(
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

    def align_ldos_to_ref(
        self,
        save_path=None,
        save_name=None,
        save_path_ext="aligned/",
        reference_index=0,
        zero_tol=1e-5,
        left_truncate=False,
        right_truncate_value=None,
        egrid_spacing_ev=0.1,
        egrid_offset_ev=-10,
        number_of_electrons=None,
        n_shift_mse=None,
    ):
        # load in the reference snapshot
        snapshot_ref = self.parameters.snapshot_directories_list[
            reference_index
        ]
        ldos_ref = np.load(
            os.path.join(
                snapshot_ref.output_npy_directory, snapshot_ref.output_npy_file
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
        emax = egrid_offset_ev + n_target * egrid_spacing_ev
        e_grid = np.linspace(
            egrid_offset_ev,
            emax,
            n_target,
            endpoint=False,
        )

        N_snapshots = len(self.parameters.snapshot_directories_list)

        for idx, snapshot in enumerate(
            self.parameters.snapshot_directories_list
        ):
            printout(f"Aligning snapshot {idx+1} of {N_snapshots}")
            ldos = np.load(
                os.path.join(
                    snapshot.output_npy_directory,
                    snapshot.output_npy_file,
                ),
                mmap_mode="r",
            )

            # get the mean
            ngrid = ldos.shape[0]
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

            e_shift = optimal_shift * egrid_spacing_ev
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
                    egrid_offset_ev
                    + (left_index_ref + optimal_shift) * egrid_spacing_ev
                )
            else:
                new_egrid_offset = egrid_offset_ev

            # reshape
            ldos_shifted = ldos_shifted.reshape(ngrid, ngrid, ngrid, -1)

            ldos_shift_info = {
                "ldos_shift_ev": round(e_shift, 4),
                "aligned_ldos_gridoffset_ev": round(new_egrid_offset, 4),
                "aligned_ldos_gridsize": np.shape(ldos_shifted)[-1],
                "aligned_ldos_gridspacing": round(egrid_spacing_ev, 4),
            }

            if number_of_electrons is not None:
                ldos_shift_info["energy_shift_from_qe_ev"] = round(
                    number_of_electrons * e_shift, 4
                )

            print(ldos_shift_info)

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
                json.dump(ldos_shift_info, f)

    def calc_optimal_ldos_shift(
        self,
        e_grid,
        ldos_mean,
        ldos_mean_ref,
        left_index,
        left_index_ref,
        n_shift_mse,
    ):
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
