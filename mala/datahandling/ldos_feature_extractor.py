import os.path

from mala import printout, get_rank, get_size
from mala.common.parallelizer import barrier, get_comm
from mala.targets.target import Target
import numpy as np
from scipy.fft import rfft, irfft
from scipy.optimize import curve_fit

import time


class LDOSFeatureExtractor:

    def __init__(
        self,
        parameters,
        target_calculator=None,
        rescaling_factor=1000,
        number_of_gaussians=8,
        maximum_number_of_retries=3,
    ):
        self.parameters = parameters
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)
        self.rescaling_factor = rescaling_factor
        self.number_of_gaussians = number_of_gaussians
        self.maximum_number_of_retries = maximum_number_of_retries

    def extract_features(
        self,
        ldos_file,
        cutoff=30,
        dry_run=False,
        file_based_communication=False,
    ):
        # Initialize everything needed for the fit.
        initial_guess_left = self.__initial_guess_linear()
        initial_guess_right = self.__initial_guess_linear()
        initial_guess_gaussian = self.__initial_guess_gaussians()

        # Extract the basename of the file
        path = os.path.dirname(ldos_file)
        filename = os.path.basename(ldos_file).split(".")[0]
        extracted_filename = os.path.join(path, filename + "_extracted.npy")
        filtered_filename = os.path.join(path, filename + "_filtered.npy")
        if file_based_communication:
            memmap = os.path.join(path, filename + "_temp.npy_")
        else:
            memmap = None

        # Load and filter the LDOS.
        if self.parameters.use_mpi:

            # This is only a quick and dirty implementation. I am splitting
            # The LDOS data across the ranks along the z-axis, for the
            # sole reason that makes sense for graphene cells.
            # This is not a general solution, and should be replaced by one in
            # the future.
            ldos_pointer = np.load(ldos_file, mmap_mode="r")
            full_ldos_shape = np.shape(ldos_pointer)

            # Every rank gets an equal number of z-planes, except for the
            # first ranks, which share the rest.
            z_planes_evenly_split = full_ldos_shape[2] // get_size()
            z_planes_unevenly_split = (
                full_ldos_shape[2] - z_planes_evenly_split * get_size()
            )

            # Assigning the rest to the first ranks.
            # It may make more sense to assign those to the middle ranks
            # in the case of bilayer graphene, but this should work for now.
            if get_rank() < z_planes_unevenly_split:
                z_start = get_rank() * (z_planes_evenly_split + 1)
                z_end = z_start + z_planes_evenly_split + 1
            else:
                z_start = (
                    get_rank() * z_planes_evenly_split
                    + z_planes_unevenly_split
                )
                z_end = z_start + z_planes_evenly_split

            ldos = ldos_pointer[:, :, z_start:z_end, :].copy()
            gridsize = (
                full_ldos_shape[0] * full_ldos_shape[1] * (z_end - z_start)
            )
            ldos = ldos.reshape(gridsize, full_ldos_shape[3])

            if dry_run:
                ldos = ldos[
                    np.random.randint(
                        0,
                        gridsize,
                        size=100,
                    ),
                    :,
                ]
                gridsize = 100
            print("Rank", get_rank(), "Doing z portions", z_start, z_end)
        else:
            ldos = self.target_calculator.read_from_numpy_file(ldos_file)
            ldos_shape = np.shape(ldos)
            gridsize = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]

            ldos = ldos.reshape(
                [
                    gridsize,
                    ldos_shape[3],
                ]
            )
            if dry_run:
                ldos = ldos[
                    np.random.randint(
                        0,
                        ldos_shape[0] * ldos_shape[1] * ldos_shape[2],
                        size=100,
                    ),
                    :,
                ]
                gridsize = 100
        # TODO: The FFT changes the dimensions from 301 to 300 - this is OK
        # for the moment since we are interested in qualitative results,
        # but should eventually be fixed!
        ldos = self.__lowpass_filter(ldos, cutoff=cutoff)
        ldos_shape = np.shape(ldos)

        if dry_run and not self.parameters.use_mpi:
            np.save(filtered_filename, ldos)

        ldos *= self.rescaling_factor
        xdata = np.arange(ldos_shape[-1])

        # 5 = 2 for left linear function, 2 for right linear function, 1 for
        # position of minimum.
        features = np.zeros(
            (gridsize, 5 + 3 * self.number_of_gaussians), dtype=ldos.dtype
        )

        # Time for-loop.
        start_time = time.perf_counter()

        # For each point within the LDOS, perform a fit to extract the
        # features.
        old_average_time = 100
        for point in range(0, gridsize):
            ydata = ldos[point, :]
            trial_counter = 0
            while trial_counter < self.maximum_number_of_retries:
                try:
                    features[point, :] = self.composite_fit(
                        xdata,
                        ydata,
                        initial_guess_left,
                        initial_guess_right,
                        initial_guess_gaussian,
                    )
                    break
                except RuntimeError:
                    trial_counter += 1
                    print(
                        "Rank",
                        get_rank(),
                        ": Fit failed at point ",
                        point,
                        ", retrying.",
                    )

                    # Retrying with different initial guess.
                    initial_guess_left = self.__initial_guess_linear()
                    initial_guess_right = self.__initial_guess_linear()
                    initial_guess_gaussian = self.__initial_guess_gaussians()

            if trial_counter == self.maximum_number_of_retries:
                print(
                    "Rank",
                    get_rank(),
                    ": Fit failed at point ",
                    self.maximum_number_of_retries,
                    " times. Zeroing out features.",
                )
                features[point, :] = self.__zero_out_features(ydata)

            if not dry_run:
                initial_guess_left = features[point, 1:3]
                initial_guess_right = features[point, 3:5]
                initial_guess_gaussian = features[point, 5:]
            if point % 100 == 0 and point != 0:
                print("Rank", get_rank(), "Point: ", point, "/", gridsize)
                average_time = (time.perf_counter() - start_time) / point
                print(
                    "Rank",
                    get_rank(),
                    ": Time per point: ",
                    average_time,
                )
                if point % 1000 == 0:
                    if average_time > old_average_time * 1.5:
                        print(
                            "Rank",
                            get_rank(),
                            ": Time per point increased. "
                            "Resetting initial value.",
                        )
                        initial_guess_left = features[point, 1:3]
                        initial_guess_right = features[point, 3:5]
                        initial_guess_gaussian = features[point, 5:]

                    old_average_time = average_time
        barrier()

        if self.parameters.use_mpi:
            features = self.__gather_ldos_features(
                features,
                full_ldos_shape,
                5 + 3 * self.number_of_gaussians,
                z_start,
                z_end,
                use_memmap=memmap,
            )

        if get_rank() == 0:
            np.save(extracted_filename, features)

            if memmap is not None:
                os.remove(memmap)

            # For debugging purposes
            # np.save(extracted_filename, ldos)

    def reconstruct_ldos(
        self, ldos_features, original_dimension=300, n_gaussians=8
    ):
        xdata = np.arange(original_dimension)
        splitting_x = int(ldos_features[0])
        x_left = slice(0, splitting_x, 1)
        x_right = slice(splitting_x, original_dimension, 1)

        predicted_left = self.linear(xdata[x_left], *ldos_features[1:3])
        predicted_right = self.linear(xdata[x_right], *ldos_features[3:5])
        predicted_twolinear = np.concatenate((predicted_left, predicted_right))
        predicted_gaussian = self.gaussians(xdata, *ldos_features[5:])

        return (
            predicted_twolinear + predicted_gaussian
        ) / self.rescaling_factor

    def __zero_out_features(self, ydata):
        splitting_x = int(np.shape(ydata)[0] / 2)
        popt_left = [0, 0]
        popt_right = [0, 0]
        popt_2 = np.concatenate(
            (
                np.zeros(self.number_of_gaussians),
                np.arange(
                    np.shape(ydata)[0] / (self.number_of_gaussians + 1),
                    np.shape(ydata)[0],
                    step=np.shape(ydata)[0] / (self.number_of_gaussians + 1),
                ),
                np.zeros(self.number_of_gaussians) + np.sqrt(0.5),
            )
        )

        return np.concatenate(([splitting_x], popt_left, popt_right, popt_2))

    def composite_fit(
        self,
        xdata,
        ydata,
        initial_guess_left,
        initial_guess_right,
        initial_guess_gaussian,
    ):
        if np.sum(ydata) == 0:
            return self.__zero_out_features(ydata)
        else:
            splitting_x = np.argmin(ydata)
            x_left = slice(0, splitting_x, 1)
            x_right = slice(splitting_x, np.shape(ydata)[0], 1)

            popt_left, pcov_left = curve_fit(
                self.linear,
                xdata[x_left],
                ydata[x_left],
                p0=initial_guess_left,
                maxfev=10000,
                method="trf",
            )
            popt_right, pcov_right = curve_fit(
                self.linear,
                xdata[x_right],
                ydata[x_right],
                p0=initial_guess_right,
                maxfev=10000,
                method="trf",
            )

            predicted_left = self.linear(xdata[x_left], *popt_left)
            predicted_right = self.linear(xdata[x_right], *popt_right)
            predicted_twolinear = np.concatenate(
                (predicted_left, predicted_right)
            )
            reduced = ydata - predicted_twolinear

            popt_2, pcov_2 = curve_fit(
                self.gaussians,
                xdata,
                reduced,
                p0=initial_guess_gaussian,
                maxfev=10000,
                method="trf",
            )

            ordering = np.argsort(
                popt_2[self.number_of_gaussians : self.number_of_gaussians * 2]
            )
            popt_2[0 : self.number_of_gaussians] = popt_2[
                0 : self.number_of_gaussians
            ][ordering]
            popt_2[self.number_of_gaussians : self.number_of_gaussians * 2] = (
                popt_2[
                    self.number_of_gaussians : self.number_of_gaussians * 2
                ][ordering]
            )
            popt_2[self.number_of_gaussians * 2 :] = popt_2[
                self.number_of_gaussians * 2 :
            ][ordering]

            # Implement error metric

            return np.concatenate(
                ([splitting_x], popt_left, popt_right, popt_2)
            )

    def linear(self, x, *params):
        params = np.array(params)
        poly_params = params
        x = x.reshape(-1, 1)
        poly = np.sum(poly_params * x ** np.arange(2), axis=-1)
        return poly

    def gaussians(self, x, *params):
        params = np.array(params)
        gauss_params = params
        weights = gauss_params[: self.number_of_gaussians]
        means = gauss_params[
            self.number_of_gaussians : 2 * self.number_of_gaussians
        ]
        sigmas = gauss_params[2 * self.number_of_gaussians :]
        x = x.reshape(-1, 1)
        gauss = np.sum(
            weights * np.exp(-((x - means) ** 2) / (2 * sigmas**2)),
            axis=-1,
        )
        return gauss

    @staticmethod
    def __lowpass_filter(ldos, cutoff):
        ldos_fft = rfft(ldos, axis=-1)
        ldos_fft[:, cutoff:] = 0
        ldos = irfft(ldos_fft, axis=-1)
        return ldos

    @staticmethod
    def __initial_guess_linear():
        lbound_poly = np.ones(2) * -10
        ubound_poly = np.ones(2) * 10
        parameter_space_dim_poly = 2
        p0 = np.random.uniform(0, 1, parameter_space_dim_poly)
        # scale to bounds
        return lbound_poly + (ubound_poly - lbound_poly) * p0

    def __initial_guess_gaussians(self):
        lbound_weights = np.ones(self.number_of_gaussians) * -100
        ubound_weights = np.ones(self.number_of_gaussians) * 100

        lbound_means = np.zeros(self.number_of_gaussians)
        ubound_means = np.ones(self.number_of_gaussians) * 300

        lbound_sigmas = np.ones(self.number_of_gaussians) * 0.01
        ubound_sigmas = np.ones(self.number_of_gaussians) * 100

        lbounds = np.concatenate([lbound_weights, lbound_means, lbound_sigmas])
        ubounds = np.concatenate([ubound_weights, ubound_means, ubound_sigmas])

        parameter_space_dim_poly = 3 * self.number_of_gaussians
        p0 = np.random.uniform(0, 1, parameter_space_dim_poly)
        # scale to bounds
        return lbounds + (ubounds - lbounds) * p0

    # TODO:
    # This is partially a code duplicate from the ldos.py file. It should be
    # refactored into a common function.
    @staticmethod
    def __gather_ldos_features(
        ldos, full_ldos_shape, feature_size, z_start, z_end, use_memmap=None
    ):
        comm = get_comm()
        barrier()

        if use_memmap is None:
            # First get the indices from all the ranks.
            indices = np.array(
                comm.gather([get_rank(), z_start, z_end], root=0)
            )
            if get_rank() == 0:
                ldos_data_full = np.empty(
                    (
                        full_ldos_shape[0],
                        full_ldos_shape[1],
                        full_ldos_shape[2],
                        feature_size,
                    ),
                    dtype=ldos.dtype,
                )
                ldos_data_full[:, :, z_start:z_end, :] = np.reshape(
                    ldos,
                    (
                        full_ldos_shape[0],
                        full_ldos_shape[1],
                        z_end - z_start,
                        feature_size,
                    ),
                )[:, :, :, :]

                # No MPI necessary for first rank. For all the others,
                # collect the buffers.
                for i in range(1, get_size()):
                    local_start = indices[i][1]
                    local_end = indices[i][2]
                    local_size = local_end - local_start
                    ldos_local = np.empty(
                        local_size
                        * full_ldos_shape[0]
                        * full_ldos_shape[1]
                        * feature_size,
                        dtype=ldos.dtype,
                    )
                    comm.Recv(ldos_local, source=i, tag=100 + i)
                    ldos_data_full[:, :, local_start:local_end, :] = (
                        np.reshape(
                            ldos_local,
                            (
                                full_ldos_shape[0],
                                full_ldos_shape[1],
                                local_size,
                                feature_size,
                            ),
                        )[:, :, :, :]
                    )
            else:
                ldos_data_full = np.empty([1, 1, 1, 1])
                comm.Send(ldos, dest=0, tag=get_rank() + 100)
            barrier()
            return ldos_data_full
        else:
            if get_rank() == 0:
                ldos_data_full = np.memmap(
                    use_memmap,
                    shape=(
                        full_ldos_shape[0],
                        full_ldos_shape[1],
                        full_ldos_shape[2],
                        feature_size,
                    ),
                    mode="w+",
                    dtype=ldos.dtype,
                )
            barrier()
            if get_rank() != 0:
                ldos_data_full = np.memmap(
                    use_memmap,
                    shape=(
                        full_ldos_shape[0],
                        full_ldos_shape[1],
                        full_ldos_shape[2],
                        feature_size,
                    ),
                    mode="r+",
                    dtype=ldos.dtype,
                )
            barrier()
            ldos_data_full[:, :, z_start:z_end, :] = np.reshape(
                ldos,
                (
                    full_ldos_shape[0],
                    full_ldos_shape[1],
                    z_end - z_start,
                    feature_size,
                ),
            )[:, :, :, :]
            return ldos_data_full
