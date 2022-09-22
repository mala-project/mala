"""Tools for analyzing a trajectory."""

from warnings import warn

from ase.io.trajectory import TrajectoryReader, Trajectory
import numpy as np
from scipy.spatial import distance

from mala.common.parameters import Parameters, ParametersDataGeneration
from mala.targets import Target


class TrajectoryAnalyzer:
    """
    Provides useful functions to analyze a trajectory.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this instance.

    trajectory : ase.io.Trajectory or string
        Trajectory or path to trajectory to be analyzed.

    target_calculator : mala.targets.target.Target
        A target calculator to calculate e.g. the RDF. If None is provided,
        one will be generated ad-hoc (recommended).
    """

    def __init__(self, parameters, trajectory, target_calculator=None):
        warn("The class TrajectoryAnalyzer is experimental. The algorithms "
             "within have been tested, but the API may still be subject to "
             "large changes.")

        self.params: ParametersDataGeneration = parameters.datageneration

        # Save or read the trajectory
        if isinstance(trajectory, TrajectoryReader):
            self.trajectory = trajectory
        elif isinstance(trajectory, str):
            self.trajectory = Trajectory(trajectory)
        else:
            raise Exception("Incompatible trajectory format provided.")

        # Create target calculator.
        self.target_calculator = target_calculator
        if target_calculator is None:
            self.target_calculator = Target(parameters)

        # Initialize variables.
        self.distance_metrics = []
        self.distance_metrics_denoised = []
        self.average_distance_equilibrated = None
        self.__saved_rdf = None
        self.first_considered_snapshot = None
        self.last_considered_snapshot = None

    def get_first_snapshot(self, equilibrated_snapshot=None,
                           distance_threshold=None):
        """
        Calculate distance metrics/first equilibrated timestep on a trajectory.

        For this step, the RDF+Cosine distance will be used as a distance
        metric. Only the first snapshot is return, all the other quantities
        can be accessed as member variables of the object calling this
        function.

        Parameters
        ----------
        equilibrated_snapshot : ase.Atoms
            An equilibrated snapshot. Will usually be read from the trajectory
            itself, but may be provided by the user if desired.

        distance_threshold : float
            Distance threshold to be used. Usually determined by analyzing
            the end of the trajectory, but may be provided by user if
            e.g. multiple trajectories are to be compared.

        Returns
        -------
        first_snapshot : int
            First snapshot for which the trajectory is equilibrated.
        """
        # First, we ned to calculate the reduced metrics for the trajectory.
        # For this, we calculate the distance between all the snapshots
        # and the last one.
        self.distance_metrics = []
        if equilibrated_snapshot is None:
            equilibrated_snapshot = self.trajectory[-1]
        for idx, step in enumerate(self.trajectory):
            self.distance_metrics.append(self.
                                         _calculate_distance_between_snapshots
                                         (equilibrated_snapshot, step, "rdf",
                                          "cosine_distance", save_rdf1=True))

        # Now, we denoise the distance metrics.
        self.distance_metrics_denoised = self.__denoise(self.distance_metrics)

        # Which snapshots are considered depends on how we denoise the
        # distance metrics.
        self.first_considered_snapshot = \
            self.params.trajectory_analysis_denoising_width
        self.last_considered_snapshot = \
            np.shape(self.distance_metrics_denoised)[0]-\
            self.params.trajectory_analysis_denoising_width
        considered_length = self.last_considered_snapshot - \
            self.first_considered_snapshot

        # Next, the average of the presumed equilibrated part is calculated,
        # and then the first N number of times teps which are below this
        # average is calculated.
        self.average_distance_equilibrated = distance_threshold
        if self.average_distance_equilibrated is None:
            self.average_distance_equilibrated = \
                np.mean(self.distance_metrics_denoised[considered_length -
                                                       int(self.params.trajectory_analysis_estimated_equilibrium * considered_length):
                                                       self.last_considered_snapshot])
        is_below = True
        counter = 0
        first_snapshot = None
        for idx, dist in enumerate(self.distance_metrics_denoised):
            if self.first_considered_snapshot <= idx \
                    <= self.last_considered_snapshot:
                if is_below:
                    counter += 1
                if dist < self.average_distance_equilibrated:
                    is_below = True
                if dist >= self.average_distance_equilibrated:
                    counter = 0
                    is_below = False
                if counter == self.params.\
                        trajectory_analysis_below_average_counter:
                    first_snapshot = idx
                    break

        print("First equilibrated timestep of trajectory is", first_snapshot)
        return first_snapshot

    def _calculate_distance_between_snapshots(self, snapshot1, snapshot2,
                                              distance_metric, reduction,
                                              save_rdf1=False):
        if distance_metric == "realspace":
            positions1 = snapshot1.get_positions()
            positions2 = snapshot2.get_positions()
            if reduction == "minimal_distance":
                result = np.amin(distance.cdist(positions1, positions2),
                                 axis=0)
                result = np.mean(result)

            elif reduction == "cosine_distance":
                number_of_atoms = snapshot1.get_number_of_atoms()
                result = distance.cosine(np.reshape(positions1,
                                                    [number_of_atoms*3]),
                                         np.reshape(positions2,
                                                    [number_of_atoms*3]))

            else:
                raise Exception("Unknown distance metric reduction.")
        elif distance_metric == "rdf":
            if save_rdf1 is True:
                if self.__saved_rdf is None:
                    self.__saved_rdf = self.target_calculator.\
                        get_radial_distribution_function(snapshot1,
                                                         method="asap3")[0]
                rdf1 = self.__saved_rdf
            else:
                rdf1 = self.target_calculator.\
                        get_radial_distribution_function(snapshot1,
                                                         method="asap3")[0]
            rdf2 = self.target_calculator.\
                get_radial_distribution_function(snapshot2,
                                                 method="asap3")[0]

            if reduction == "minimal_distance":
                raise Exception("Combination of distance metric and reduction "
                                "not supported.")

            elif reduction == "cosine_distance":
                result = distance.cosine(rdf1, rdf2)

            else:
                raise Exception("Unknown distance metric reduction.")
        else:
            raise Exception("Unknown distance metric selected.")

        return result

    def __denoise(self, signal):
        denoised_signal = np.convolve(signal, np.ones(
            self.params.trajectory_analysis_denoising_width)
                                      / self.params.
                                      trajectory_analysis_denoising_width,
                                      mode='same')
        return denoised_signal

