import os.path

from mala import printout
from mala.targets.target import Target
import numpy as np
from scipy.fft import rfft, irfft
from scipy.optimize import curve_fit

import time


class LDOSFeatureExtractor:

    def __init__(self, parameters, target_calculator=None):
        self.parameters = parameters
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)

    def extract_features(self, ldos_file, cutoff=30, n_gaussians=8):
        # Initialize everything needed for the fit.
        initial_guess_left = self.__initial_guess_linear()
        initial_guess_right = self.__initial_guess_linear()
        initial_guess_gaussian = self.__initial_guess_gaussians(n_gaussians)

        # Extract the basename of the file
        path = os.path.dirname(ldos_file)
        filename = os.path.basename(ldos_file)
        extracted_filename = os.path.join(path, filename, "_extracted.npy")

        # Load and filter the LDOS.
        ldos = self.target_calculator.read_from_numpy_file(ldos_file)[
            :40, :40, :80, :
        ]
        # TODO: The FFT changes the dimensions from 301 to 300 - this is OK
        # for the moment since we are interested in qualitative results,
        # but should eventually be fixed!
        ldos = self.__lowpass_filter(ldos, cutoff=cutoff)
        ldos_shape = np.shape(ldos)
        ldos = ldos.reshape(
            [
                ldos_shape[0] * ldos_shape[1] * ldos_shape[2],
                ldos_shape[3],
            ]
        )
        gridsize = ldos_shape[0] * ldos_shape[1] * ldos_shape[2]
        xdata = np.arange(ldos_shape[-1])

        # 5 = 2 for left linear function, 2 for right linear function, 1 for
        # position of minimum.
        features = np.zeros((gridsize, 5 + 3 * n_gaussians), dtype=ldos.dtype)

        # Time for-loop.
        start_time = time.perf_counter()

        # For each point within the LDOS, perform a fit to extract the
        # features.
        for point in range(0, gridsize):
            ydata = ldos[point, :]
            trial_counter = 0
            while trial_counter < 5:
                try:
                    features[point, :] = self.composite_fit(
                        xdata,
                        ydata,
                        initial_guess_left,
                        initial_guess_right,
                        initial_guess_gaussian,
                        n_gaussians,
                    )
                    break
                except RuntimeError:
                    trial_counter += 1
                    printout("Fit failed at point ", point, ", retrying.")
            if trial_counter == 3:
                raise Exception("Fit failed at point ", point, "three times.")

            initial_guess_left = features[point, 1:3]
            initial_guess_right = features[point, 3:5]
            initial_guess_gaussian = features[point, 5:]
            if point % 100 == 0:
                print("Point: ", point, "/", gridsize)
                print(
                    "Time per point: ",
                    (time.perf_counter() - start_time) / (point + 1),
                )

    @staticmethod
    def composite_fit(
        xdata,
        ydata,
        initial_guess_left,
        initial_guess_right,
        initial_guess_gaussian,
        n_gaussians,
    ):
        def linear(x, *params):
            params = np.array(params)
            poly_params = params
            x = x.reshape(-1, 1)
            poly = np.sum(poly_params * x ** np.arange(2), axis=-1)
            return poly

        def gaussians(x, *params):
            params = np.array(params)
            gauss_params = params
            weights = gauss_params[:n_gaussians]
            means = gauss_params[n_gaussians : 2 * n_gaussians]
            sigmas = gauss_params[2 * n_gaussians :]
            x = x.reshape(-1, 1)
            gauss = np.sum(
                weights * np.exp(-((x - means) ** 2) / (2 * sigmas**2)),
                axis=-1,
            )
            return gauss

        def gaussians_error(x, *params):
            return gaussians(x, *params) - ydata

        if np.sum(ydata) == 0:
            splitting_x = int(np.shape(ydata)[0] / 2)
            popt_left = [0, 0]
            popt_right = [0, 0]
            popt_2 = np.concatenate(
                (
                    np.zeros(n_gaussians),
                    np.arange(
                        np.shape(ydata)[0] / (n_gaussians + 1),
                        np.shape(ydata)[0],
                        step=np.shape(ydata)[0] / (n_gaussians + 1),
                    ),
                    np.zeros(n_gaussians) + np.sqrt(0.5),
                )
            )

        else:
            splitting_x = np.argmin(ydata)
            x_left = slice(0, splitting_x, 1)
            x_right = slice(splitting_x, np.shape(ydata)[0], 1)

            popt_left, pcov_left = curve_fit(
                linear,
                xdata[x_left],
                ydata[x_left],
                p0=initial_guess_left,
                # bounds=bounds,
                maxfev=10000,
                method="trf",
            )
            popt_right, pcov_right = curve_fit(
                linear,
                xdata[x_right],
                ydata[x_right],
                p0=initial_guess_right,
                # bounds=bounds,
                maxfev=10000,
                method="trf",
            )

            predicted_left = linear(xdata[x_left], *popt_left)
            predicted_right = linear(xdata[x_right], *popt_right)
            predicted_twolinear = np.concatenate(
                (predicted_left, predicted_right)
            )
            reduced = ydata - predicted_twolinear

            popt_2, pcov_2 = curve_fit(
                gaussians_error,
                xdata,
                reduced,
                p0=initial_guess_gaussian,
                maxfev=10000,
                method="trf",
            )

            ordering = np.argsort(popt_2[n_gaussians : n_gaussians * 2])
            popt_2[0:n_gaussians] = popt_2[0:n_gaussians][ordering]
            popt_2[n_gaussians : n_gaussians * 2] = popt_2[
                n_gaussians : n_gaussians * 2
            ][ordering]
            popt_2[n_gaussians * 2 :] = popt_2[n_gaussians * 2 :][ordering]

            # Implement error metric

        return np.concatenate(([splitting_x], popt_left, popt_right, popt_2))

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

    @staticmethod
    def __initial_guess_gaussians(n_gaussians):
        lbound_weights = np.ones(n_gaussians) * -100
        ubound_weights = np.ones(n_gaussians) * 100

        lbound_means = np.zeros(n_gaussians)
        ubound_means = np.ones(n_gaussians) * 300

        lbound_sigmas = np.ones(n_gaussians) * 0.01
        ubound_sigmas = np.ones(n_gaussians) * 100

        lbounds = np.concatenate([lbound_weights, lbound_means, lbound_sigmas])
        ubounds = np.concatenate([ubound_weights, ubound_means, ubound_sigmas])

        parameter_space_dim_poly = 3 * n_gaussians
        p0 = np.random.uniform(0, 1, parameter_space_dim_poly)
        # scale to bounds
        return lbounds + (ubounds - lbounds) * p0
