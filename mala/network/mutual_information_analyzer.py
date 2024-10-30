"""Class for performing a full mutual information analysis."""

import itertools
import os

import numpy as np

from mala.datahandling.data_converter import (
    descriptor_input_types,
    target_input_types,
)
from mala.network.acsd_analyzer import ACSDAnalyzer
import sklearn.mixture
import sklearn.covariance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

descriptor_input_types_acsd = descriptor_input_types + ["numpy", "openpmd"]
target_input_types_acsd = target_input_types + ["numpy", "openpmd"]


class MutualInformationAnalyzer(ACSDAnalyzer):
    """
    Analyzer based on mutual information analysis.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        The descriptor calculator used for parsing/converting fingerprint
        data. If None, the descriptor calculator will be created by this
        object using the parameters provided. Default: None

    target_calculator : mala.targets.target.Target
        Target calculator used for parsing/converting target data. If None,
        the target calculator will be created by this object using the
        parameters provided. Default: None
    """

    def __init__(
        self, params, target_calculator=None, descriptor_calculator=None
    ):
        super(MutualInformationAnalyzer, self).__init__(
            params,
            target_calculator=target_calculator,
            descriptor_calculator=descriptor_calculator,
        )

    @staticmethod
    def __get_gmm(
        data,
        n_components=48,
        max_iter=1000,
        covariance_type="diag",
        reg_covar=1e-6,
    ):
        gmm = sklearn.mixture.GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
        )
        gmm.fit(data)
        return gmm

    @staticmethod
    def _calculate_acsd(
        descriptor_data,
        ldos_data,
        acsd_points,
        descriptor_vectors_contain_xyz=True,
    ):
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
        descriptor_dim = np.shape(descriptor_data)
        ldos_dim = np.shape(ldos_data)
        if len(descriptor_dim) == 4:
            descriptor_data = np.reshape(
                descriptor_data,
                (
                    descriptor_dim[0] * descriptor_dim[1] * descriptor_dim[2],
                    descriptor_dim[3],
                ),
            )
            if descriptor_vectors_contain_xyz:
                descriptor_data = descriptor_data[:, 3:]
        elif len(descriptor_dim) != 2:
            raise Exception("Cannot work with this descriptor data.")

        if len(ldos_dim) == 4:
            ldos_data = np.reshape(
                ldos_data,
                (ldos_dim[0] * ldos_dim[1] * ldos_dim[2], ldos_dim[3]),
            )
        elif len(ldos_dim) != 2:
            raise Exception("Cannot work with this LDOS data.")

        n_components = 48
        max_iter = 1000
        rand_perm = np.random.permutation(ldos_data.shape[0])
        perm_train = rand_perm[:acsd_points]
        X_train = descriptor_data[perm_train]
        Y_train = ldos_data[perm_train]
        covariance = sklearn.covariance.EmpiricalCovariance()
        covariance.fit(Y_train)
        plot = False
        if plot:
            plt.imshow(
                covariance.covariance_, cmap="cool", interpolation="nearest"
            )
            plt.show()

        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        covariance = sklearn.covariance.EmpiricalCovariance()
        covariance.fit(X_train)
        if plot:
            plt.imshow(
                covariance.covariance_, cmap="hot", interpolation="nearest"
            )
            plt.show()

        X_shuffled = X_train[np.random.permutation(X_train.shape[0])]

        combined_train = np.concatenate((X_train, Y_train), axis=1)
        combined_shuffled = np.concatenate((X_shuffled, Y_train), axis=1)

        # Apply a variance filter
        variances_ldos = np.var(Y_train, axis=0)
        variances_bisprectrum = np.var(X_train, axis=0)
        filtered = np.shape(variances_bisprectrum > 1.0)

        gmm_Y = MutualInformationAnalyzer.__get_gmm(
            Y_train,
            n_components=int(ldos_dim[0] / 2),
            covariance_type="diag",
        )

        gmm_X = MutualInformationAnalyzer.__get_gmm(
            X_train,
            n_components=int(descriptor_dim[0] / 2),
            covariance_type="diag",
            reg_covar=1e-5,
        )

        log_p_X = gmm_X.score_samples(X_train)
        log_p_Y = gmm_Y.score_samples(Y_train)

        gmm_combined = MutualInformationAnalyzer.__get_gmm(
            combined_train,
            n_components=int(descriptor_dim[0] / 2),
            max_iter=max_iter,
            covariance_type="diag",
        )
        gmm_combined_shuffled = MutualInformationAnalyzer.__get_gmm(
            combined_shuffled,
            n_components=int(descriptor_dim[0] / 2),
            max_iter=max_iter,
            covariance_type="diag",
        )

        log_p_combined = gmm_combined.score_samples(combined_train)
        mi = np.mean(log_p_combined) - np.mean(log_p_Y) - np.mean(log_p_X)

        log_p_shuffled = gmm_combined_shuffled.score_samples(combined_shuffled)
        mi_shuffled = (
            np.mean(log_p_shuffled) - np.mean(log_p_Y) - np.mean(log_p_X)
        )

        mi_difference = mi - mi_shuffled
        return mi_difference
