"""Class for performing a full ACSD analysis."""

import numpy as np

from mala.datahandling.data_converter import (
    descriptor_input_types,
    target_input_types,
)
from mala.network.descriptor_scoring_optimizer import (
    DescriptorScoringOptimizer,
)


class ACSDAnalyzer(DescriptorScoringOptimizer):
    """
    Analyzer based on the ACSD analysis.

    Mathematical details see here: doi.org/10.1088/2632-2153/ac9956

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
        super(ACSDAnalyzer, self).__init__(
            params,
            target_calculator=target_calculator,
            descriptor_calculator=descriptor_calculator,
        )

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
        elif score < self.best_score:
            self.best_score = score
            self.best_trial_index = index

    def _get_best_trial(self):
        """Determine the best trial as given by this study."""
        return self.averaged_study[
            np.argmin(np.float64(np.array(self.averaged_study)[:, -1]))
        ]

    @staticmethod
    def _calculate_cosine_similarities(
        descriptor_data,
        ldos_data,
        nr_points,
        descriptor_vectors_contain_xyz=True,
    ):
        """
        Calculate the raw cosine similarities for descriptor and LDOS data.

        Parameters
        ----------
        descriptor_data : numpy.ndarray
            Array containing the descriptors.

        ldos_data : numpy.ndarray
            Array containing the LDOS.

        descriptor_vectors_contain_xyz : bool
            If true, the xyz values are cut from the beginning of the
            descriptor vectors.

        nr_points : int
            The number of points for which to calculate the ACSD.
            The actual number of distances will be acsd_points x acsd_points,
            since the cosine similarity is only defined for pairs.

        Returns
        -------
        similarity_array : numpy.ndarray
            A (2,nr_points*nr_points) array containing the cosine similarities.

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

        similarity_array = []
        # Draw nr_points at random from snapshot.
        rng = np.random.default_rng()
        points_i = rng.choice(
            np.shape(descriptor_data)[0],
            size=np.shape(descriptor_data)[0],
            replace=False,
        )
        for i in range(0, nr_points):
            # Draw another nr_points at random from snapshot.
            rng = np.random.default_rng()
            points_j = rng.choice(
                np.shape(descriptor_data)[0],
                size=np.shape(descriptor_data)[0],
                replace=False,
            )

            for j in range(0, nr_points):
                # Calculate similarities between these two pairs.
                descriptor_distance = ACSDAnalyzer.__calc_cosine_similarity(
                    descriptor_data[points_i[i]], descriptor_data[points_j[j]]
                )
                ldos_distance = ACSDAnalyzer.__calc_cosine_similarity(
                    ldos_data[points_i[i]], ldos_data[points_j[j]]
                )
                similarity_array.append([descriptor_distance, ldos_distance])

        return np.array(similarity_array)

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
        return self._calculate_acsd(
            descriptor,
            target,
            self.params.hyperparameters.acsd_points,
            descriptor_vectors_contain_xyz=self.params.descriptors.descriptors_contain_xyz,
        )

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
        similarity_data = ACSDAnalyzer._calculate_cosine_similarities(
            descriptor_data,
            ldos_data,
            acsd_points,
            descriptor_vectors_contain_xyz=descriptor_vectors_contain_xyz,
        )
        data_size = np.shape(similarity_data)[0]

        # Subtracting LDOS similarities from bispectrum similiarities.
        distances = similarity_data[:, 0] - similarity_data[:, 1]
        distances = distances.clip(min=0)
        return np.mean(distances)

    @staticmethod
    def __calc_cosine_similarity(vector1, vector2, norm=2):
        """
        Calculate the cosine similarity between two vectors.

        The type of norm which is used can be adapted. Cosine similarity
        always lies between 0 (not similar) and 1 (similar).

        Parameters
        ----------
        vector1 : numpy.ndarray
            The first vector.

        vector2 : numpy.ndarray
            The second vector.

        norm : int
            The order of the norm to be used. Default is 2.

        Returns
        -------
        similarity : float
            The cosine similarity between the two vectors.
        """
        if np.shape(vector1)[0] != np.shape(vector2)[0]:
            raise Exception(
                "Cannot calculate similarity between vectors "
                "of different dimenstions."
            )
        if np.shape(vector1)[0] == 1:
            return np.min([vector1[0], vector2[0]]) / np.max(
                [vector1[0], vector2[0]]
            )
        else:
            return np.dot(vector1, vector2) / (
                np.linalg.norm(vector1, ord=norm)
                * np.linalg.norm(vector2, ord=norm)
            )
