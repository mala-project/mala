import os.path

from mala.targets.target import Target
import numpy as np
from scipy.fft import rfft, irfft
from scipy.optimize import curve_fit


class LDOSFeatureExtractor:

    def __init__(self, parameters, target_calculator=None):
        self.parameters = parameters
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(parameters)

    def extract_features(
        self, ldos_file, cutoff=30, poly_order=3, n_gaussians=8
    ):
        # Initialize everything needed for the fit.
        p0_poly = self.__init_poly(poly_order)
        p0_gaussian = self.__init_gaussians(n_gaussians)

        # Extract the basename of the file
        path = os.path.dirname(ldos_file)
        filename = os.path.basename(ldos_file)
        extracted_filename = os.path.join(path, filename, "_extracted.npy")

        # Load and filter the LDOS.
        ldos = self.target_calculator.read_from_numpy_file(ldos_file)
        ldos = ldos.reshape((-1, ldos.shape[-1]))
        ldos = self.__lowpass_filter(ldos, cutoff=cutoff)
        gridsize = ldos.shape[-1]

        # For each point within the LDOS, perform a fit to extract the
        # features.
        xdata = np.arange(ldos.shape[-1])
        features = np.zeros(
            (gridsize, poly_order + 1 + 3 * n_gaussians), dtype=ldos.dtype
        )
        for point in range(0, gridsize):
            ydata = ldos[point]
            features[point, :] = self.composite_fit(
                xdata, ydata, p0_poly, p0_gaussian, poly_order, n_gaussians
            )

    @staticmethod
    def composite_fit(
        xdata, ydata, p0_poly, p0_gaussian, poly_order, n_gaussians
    ):
        def poly(x, *params):
            params = np.array(params)
            poly_params = params
            x = x.reshape(-1, 1)
            poly = np.sum(
                poly_params * x ** np.arange(poly_order + 1), axis=-1
            )
            return poly

        def poly_error(x, *params):
            return poly(x, *params) - ydata

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
            return gauss - ydata

        def gaussians_error(x, *params):
            return gaussians(x, *params) - ydata

        # print(xdata.shape, ydata.shape, p0.shape)
        popt_1, pcov_1 = curve_fit(
            poly_error,
            xdata,
            ydata,
            p0=p0_poly,
            maxfev=10000,
            method="trf",
        )

        predicted_poly = poly(xdata, *popt_1)
        reduced = ydata - predicted_poly

        popt_2, pcov_2 = curve_fit(
            gaussians_error,
            xdata,
            reduced,
            p0=p0_gaussian,
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

        return popt_1, popt_2

    @staticmethod
    def __lowpass_filter(ldos, cutoff):
        ldos_fft = rfft(ldos, axis=-1)
        ldos_fft[:, cutoff:] = 0
        ldos = irfft(ldos_fft, axis=-1)
        return ldos

    @staticmethod
    def __init_poly(poly_order):
        lbound_poly = np.ones(poly_order + 1) * -10
        ubound_poly = np.ones(poly_order + 1) * 10
        parameter_space_dim_poly = poly_order + 1
        p0 = np.random.uniform(0, 1, parameter_space_dim_poly)
        # scale to bounds
        return lbound_poly + (ubound_poly - lbound_poly) * p0

    @staticmethod
    def __init_gaussians(n_gaussians):
        lbound_weights = np.zeros(n_gaussians)
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
