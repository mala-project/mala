"""Class for performing a full mutual information analysis."""

import numpy as np


from mala.common.parallelizer import parallel_warn
from mala.network.descriptor_scoring_optimizer import (
    DescriptorScoringOptimizer,
)


class MutualInformationAnalyzer(DescriptorScoringOptimizer):
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
        parallel_warn(
            "The MutualInformationAnalyzer is still in its "
            "experimental stage. The API is consistent with "
            "MALA hyperparameter optimization and will likely not "
            "change, but the internal algorithm may be subject "
            "to changes in the near-future."
        )
        super(MutualInformationAnalyzer, self).__init__(
            params,
            target_calculator=target_calculator,
            descriptor_calculator=descriptor_calculator,
        )

    def _get_best_trial(self):
        """Determine the best trial as given by this study."""
        return self.averaged_study[
            np.argmax(np.array(self.averaged_study[:, -1]))
        ]

    def _update_logging(self, score, index):
        """
        Update logging information with current score and index.

        Parameters
        ----------
        score : float
            The score of the current trial.

        index :
            The index of the current trial.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_trial_index = index
        elif score > self.best_score:
            self.best_score = score
            self.best_trial_index = index

    def _calculate_score(self, descriptor, target):
        """
        Calculate score of a descriptor/target pair.

        Parameters
        ----------
        descriptor : numpy.ndarray
            The descriptor data.

        target : numpy.ndarray
            The target data.

        Returns
        -------
        score : float
            The score of this descriptor/target pair.
        """
        return self._calculate_mutual_information(
            descriptor,
            target,
            self.params.hyperparameters.mutual_information_points,
            descriptor_vectors_contain_xyz=self.params.descriptors.descriptors_contain_xyz,
        )

    @staticmethod
    def _calculate_mutual_information(
        descriptor_data,
        ldos_data,
        n_samples,
        descriptor_vectors_contain_xyz=True,
    ):
        """
        Calculate the Mutual Information for given descriptor and LDOS data.

        Mutual Information measures how well the descriptors capture the
        relevant information that is needed to predict the LDOS.
        The unit of MI is bits.

        Parameters
        ----------
        descriptor_data : numpy.ndarray
            Array containing the descriptors.

        ldos_data : numpy.ndarray
            Array containing the LDOS.

        n_samples : int
            The number of points for which to calculate the mutual information.

        Returns
        -------
        mi : float
            The mutual information between the descriptor and the LDOS in bits.

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

        # The hyperparameters could be put potentially into the params.
        mi = MutualInformationAnalyzer._mutual_information(
            descriptor_data,
            ldos_data,
            n_components=None,
            n_samples=n_samples,
            covariance_type="diag",
            normalize_data=True,
        )
        return mi

    @staticmethod
    def _normalize(data):
        """
        Normalize the data to have zero mean and unit variance.

        The normalization is done in-place.

        Parameters
        ----------
        data : numpy.ndarray
            The data to be normalized.

        Returns
        -------
        data : numpy.ndarray
            The normalized data.
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std_nonzero = std > 1e-6
        data = data[:, std_nonzero]
        mean = mean[std_nonzero]
        std = std[std_nonzero]
        data = (data - mean) / std
        return data

    @staticmethod
    def _mutual_information(
        X,
        Y,
        n_components=None,
        max_iter=1000,
        n_samples=100000,
        covariance_type="diag",
        normalize_data=False,
    ):
        """
        Calculate the Mutual Information for given input (X) and output (Y).

        Mutual information is calculated by fitting Gaussian mixture models to
        combined and individual data distributions.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Y : numpy.ndarray
            Output data.

        n_components : int
            Number of components in the Gaussian mixture model.

        max_iter : int
            Maximum number of iterations for the Gaussian mixture model.

        n_samples : int
            Number of samples to use for the calculation.

        covariance_type : str
            Type of covariance matrix to use in the Gaussian mixture model.
            Currently, only "diag" is supported.

        normalize_data : bool
            If True, the data is normalized to have zero mean and unit
            variance. This can help with robustness.

        Returns
        -------
        mi : float
            The mutual information between the input and output data.
        """
        import sklearn.mixture
        import sklearn.covariance

        assert (
            covariance_type == "diag"
        ), "Only support covariance_type='diag' for now"
        n = X.shape[0]
        dim_X = X.shape[-1]
        rand_subset = np.random.permutation(n)[:n_samples]
        if normalize_data:
            X = MutualInformationAnalyzer._normalize(X)
            Y = MutualInformationAnalyzer._normalize(Y)
        X = X[rand_subset]
        Y = Y[rand_subset]
        XY = np.concatenate([X, Y], axis=1)
        d = XY.shape[-1]
        if n_components is None:
            n_components = d // 2
        gmm_XY = sklearn.mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
        )
        gmm_XY.fit(XY)

        gmm_X = sklearn.mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
        )
        gmm_X.weights_ = gmm_XY.weights_
        gmm_X.means_ = gmm_XY.means_[:, :dim_X]
        gmm_X.covariances_ = gmm_XY.covariances_[:, :dim_X]
        gmm_X.precisions_ = gmm_XY.precisions_[:, :dim_X]
        gmm_X.precisions_cholesky_ = gmm_XY.precisions_cholesky_[:, :dim_X]

        gmm_Y = sklearn.mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
        )
        gmm_Y.weights_ = gmm_XY.weights_
        gmm_Y.means_ = gmm_XY.means_[:, dim_X:]
        gmm_Y.covariances_ = gmm_XY.covariances_[:, dim_X:]
        gmm_Y.precisions_ = gmm_XY.precisions_[:, dim_X:]
        gmm_Y.precisions_cholesky_ = gmm_XY.precisions_cholesky_[:, dim_X:]

        rand_perm = np.random.permutation(Y.shape[0])
        Y_perm = Y[rand_perm]
        XY_perm = np.concatenate([X, Y_perm], axis=1)
        temp = (
            gmm_XY.score_samples(XY_perm)
            - gmm_X.score_samples(X)
            - gmm_Y.score_samples(Y_perm)
        )
        temp_exp = np.exp(temp)
        mi = np.mean(temp_exp * temp)
        # change log base e to log base 2
        mi = mi / np.log(2)
        return mi
