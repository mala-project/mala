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
        use_feature_scaling=True,
    ):
        self.parameters = parameters
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)
        self.rescaling_factor = rescaling_factor
        self.number_of_gaussians = number_of_gaussians
        self.maximum_number_of_retries = maximum_number_of_retries
        self.use_feature_scaling = use_feature_scaling

    def extract_features(
        self,
        ldos_file,
        cutoff=30,
        dry_run=False,
        file_based_communication=False,
    ):
        dry_run_size = 100

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
                        size=dry_run_size,
                    ),
                    :,
                ]
                gridsize = dry_run_size
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
                        size=dry_run_size,
                    ),
                    :,
                ]
                gridsize = dry_run_size
        # TODO: The FFT changes the dimensions from 301 to 300 - this is OK
        # for the moment since we are interested in qualitative results,
        # but should eventually be fixed!
        ldos = self.__lowpass_filter(ldos, cutoff=cutoff)
        ldos_shape = np.shape(ldos)

        if dry_run and not self.parameters.use_mpi:
            np.save(filtered_filename, ldos)

        ldos *= self.rescaling_factor

        # Initialize everything needed for the fit.
        self.__initialize_bound_gaussian()
        initial_guess_left = self.__initial_guess_linear()
        initial_guess_right = self.__initial_guess_linear()
        initial_guess_gaussian = self.__initial_guess_gaussians()
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

                    # VERY rarely, one of the weights gets set to infinity
                    # without the fit failing. This should not happen, but
                    # we can catch it here. Eventually we should maybe find
                    # out why this happens.
                    if np.any(np.isinf(features[point, :])):
                        self.__zero_out_features(ydata)
                    break
                except (RuntimeError, ValueError):
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
            if point % 100 == 0:
                print("Rank", get_rank(), "Point: ", point, "/", gridsize)
                print(
                    "Rank",
                    get_rank(),
                    ": Time per point: ",
                    (time.perf_counter() - start_time) / (point + 1),
                )
        if self.use_feature_scaling:
            for f in range(5 + 3 * self.number_of_gaussians):
                features[:, f] = self.__internal_feature_scaling(
                    features[:, f], f
                )

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

    def reconstruct_single_ldos(self, ldos_features, original_dimension=300):
        xdata = np.arange(original_dimension)
        splitting_x = int(ldos_features[0])
        x_left = slice(0, splitting_x, 1)
        x_right = slice(splitting_x, original_dimension, 1)

        predicted_left = self.linear(xdata[x_left], *ldos_features[1:3])
        predicted_right = self.linear(xdata[x_right], *ldos_features[3:5])
        predicted_twolinear = np.concatenate((predicted_left, predicted_right))
        predicted_gaussian = self.gaussians(xdata, *ldos_features[5:])

        if self.use_feature_scaling:
            temp = (
                predicted_twolinear + predicted_gaussian
            ) / self.rescaling_factor
            for f in range(5 + 3 * self.number_of_gaussians):
                temp = self.__internal_feature_scaling(temp, f)
            return temp
        else:
            return (
                predicted_twolinear + predicted_gaussian
            ) / self.rescaling_factor

    def reconstruct_ldos(self, ldos_features, original_dimension=300):
        xdata = np.arange(original_dimension)
        ldos_shape = np.shape(ldos_features)

        # If we have 4D data, we need to reshape it to 2D.
        x_original = None
        y_original = None
        z_original = None
        if len(ldos_shape) == 4:
            x_original = ldos_shape[0]
            y_original = ldos_shape[1]
            z_original = ldos_shape[2]
            ldos_features = ldos_features.reshape(
                (ldos_shape[0] * ldos_shape[1] * ldos_shape[2], ldos_shape[3])
            )
        elif len(ldos_shape) != 2:
            raise Exception("Cannot work with provided LDOS shape.")
        full_ldos = np.zeros((ldos_features.shape[0], original_dimension))

        # Initializing the arrays for the left and right side of linear fit as
        # ragged arrays.
        splitting_x = np.int32(ldos_features[:, 0])
        zeros = np.zeros(ldos_features.shape[0])
        ends = np.zeros(ldos_features.shape[0]) + original_dimension
        for i, (start, stop) in enumerate(zip(zeros, splitting_x)):
            x_left = np.int32(np.arange(start, stop))
            full_ldos[i, x_left] = self.linear(
                xdata[x_left], *ldos_features[i, 1:3]
            )

        for i, (start, stop) in enumerate(zip(splitting_x, ends)):
            x_right = np.int32(np.arange(start, stop))
            full_ldos[i, x_right] = self.linear(
                xdata[x_right], *ldos_features[i, 3:5]
            )

        # Excplicitly freeing up memory.
        del splitting_x, zeros, ends

        # I haven't found a nice way yet to make the Gaussian prediction
        # work with the same function for both arrays and single values,
        # so for now, we will use two functions.
        full_ldos += self.gaussians_vector(
            np.broadcast_to(
                xdata, (ldos_features.shape[0], original_dimension)
            ),
            *ldos_features[:, 5:]
        )

        # Rescaling and reshaping, if needed.
        full_ldos /= self.rescaling_factor
        if x_original is not None:
            full_ldos = full_ldos.reshape(
                (x_original, y_original, z_original, original_dimension)
            )

        return full_ldos

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

    @staticmethod
    def __internal_feature_scaling(input_array, i):
        temp_array = input_array.copy().astype(np.float32)
        if i == 0:
            return temp_array
        if 1 <= i <= 4:
            return np.log(np.abs(temp_array))
        elif 4 < i < 13:
            return np.cbrt(temp_array)
        elif 12 < i < 21:
            return temp_array
        elif i > 20:
            return np.log(temp_array)

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

            ubound_this_point = self.ubounds.copy()
            ubound_this_point[0 : self.number_of_gaussians] *= (
                np.max(reduced) * 2.0
            )
            lbound_this_point = self.lbounds.copy()
            lbound_this_point[0 : self.number_of_gaussians] *= (
                np.max(reduced) * 2.0
            )
            # Rescaling the initial guess, IF necessary.
            if np.any(
                np.abs(initial_guess_gaussian[0 : self.number_of_gaussians])
                >= ubound_this_point[0 : self.number_of_gaussians]
            ):
                print(
                    "Rank", get_rank(), ": Initial guess too large, rescaling."
                )
                initial_guess_gaussian[0 : self.number_of_gaussians] /= np.max(
                    initial_guess_gaussian[0 : self.number_of_gaussians]
                )
                initial_guess_gaussian[0 : self.number_of_gaussians] *= np.max(
                    reduced
                )
            popt_2, pcov_2 = curve_fit(
                self.gaussians,
                xdata,
                reduced,
                p0=initial_guess_gaussian,
                maxfev=10000,
                method="trf",
                bounds=(lbound_this_point, ubound_this_point),
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

    def gaussians_vector(self, x, *params):
        params = np.array(params)
        gauss_params = params
        weights = gauss_params[:, : self.number_of_gaussians]
        means = gauss_params[
            :, self.number_of_gaussians : 2 * self.number_of_gaussians
        ]
        sigmas = gauss_params[:, 2 * self.number_of_gaussians :]
        gauss_before_sum = weights[:, None, :] * np.exp(
            -((x[:, :, None] - means[:, None, :]) ** 2)
            / (2 * sigmas[:, None, :] ** 2)
        )
        gauss = np.sum(
            gauss_before_sum,
            axis=-1,  # Sum over the data_dimension axis
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

    def __initialize_bound_gaussian(self):
        lbound_weights = (
            np.ones(self.number_of_gaussians) * -1.0 * self.rescaling_factor
        )
        ubound_weights = (
            np.ones(self.number_of_gaussians) * 1.0 * self.rescaling_factor
        )

        lbound_means = np.zeros(self.number_of_gaussians)
        ubound_means = np.ones(self.number_of_gaussians) * 500

        lbound_sigmas = np.ones(self.number_of_gaussians) * 0.001
        ubound_sigmas = np.ones(self.number_of_gaussians) * 100

        self.lbounds = np.concatenate(
            [lbound_weights, lbound_means, lbound_sigmas]
        )
        self.ubounds = np.concatenate(
            [ubound_weights, ubound_means, ubound_sigmas]
        )

    def __initial_guess_gaussians(self):

        parameter_space_dim_poly = 3 * self.number_of_gaussians
        p0 = np.random.uniform(0, 1, parameter_space_dim_poly)
        # scale to bounds
        return self.lbounds + (self.ubounds - self.lbounds) * p0

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
