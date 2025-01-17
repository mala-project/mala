"""Tools for analyzing a trajectory."""

from functools import cached_property
import os
from warnings import warn

from ase.io.trajectory import TrajectoryReader, Trajectory, TrajectoryWriter
import numpy as np
from scipy.spatial import distance

from mala.common.parameters import ParametersDataGeneration
from mala.common.parallelizer import printout
from mala.targets import Target
from mala.descriptors.descriptor import Descriptor


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

    temperatures : string or numpy.ndarray
        Array holding the temperatures for the trajectory or path to numpy
        file containing temperatures.

    target_temperature : float
        Target temperature for equilibration.

    malada_compatability : bool
        If True, twice the radius set by the minimum imaging convention (MIC)
        will be used for RDF calculation. This is generally discouraged,
        but some older malada calculations have been performed with it, so
        this parameter provides reproducibility.

    Attributes
    ----------
    parameters : mala.common.parameters.ParametersDataGeneration
        MALA data generation parameters.

    average_distance_equilibrated : float
        Distance threshold for determination of first equilibrated snapshot.

    distance_metrics_denoised : numpy.ndarray
        RDF based distance metrics used for equilibration analysis.

    distances_realspace : numpy.ndarray
        Realspace distance metrics used to sample snapshots.

    first_considered_snapshot : int
        First snapshot to be considered during equilibration analysis (i.e.,
        after pruning).

    last_considered_snapshot : int
        Last snapshot to be considered during equilibration analysis (i.e.,
        after pruning).

    target_calculator : mala.targets.target.Target
        Target calculator used for computing RDFs.
    """

    def __init__(
        self,
        parameters,
        trajectory,
        temperatures=None,
        target_calculator=None,
        target_temperature=None,
        malada_compatability=False,
    ):
        warn(
            "The class TrajectoryAnalyzer is experimental. The algorithms "
            "within have been tested, but the API may still be subject to "
            "large changes."
        )

        self.parameters: ParametersDataGeneration = parameters.datageneration

        # If needed, read the trajectory
        self.trajectory = None
        if isinstance(trajectory, TrajectoryReader):
            self.trajectory = trajectory
        elif isinstance(trajectory, str):
            self.trajectory = Trajectory(trajectory)
        else:
            raise Exception("Incompatible trajectory format provided.")

        # If needed, read the temperature files
        self._temperatures = None
        if temperatures is not None:
            if isinstance(temperatures, np.ndarray):
                self._temperatures = temperatures
            elif isinstance(temperatures, str):
                self._temperatures = np.load(temperatures)
            else:
                raise Exception("Incompatible temperature format provided.")

        # Create target calculator.
        self.target_calculator = target_calculator
        if target_calculator is None:
            self.target_calculator = Target(parameters)

        if target_temperature is not None:
            self.target_calculator.temperature = target_temperature

        # Initialize variables.
        self._distance_metrics = []
        self.distance_metrics_denoised = []
        self.average_distance_equilibrated = None
        self.__saved_rdf = None
        self.first_considered_snapshot = None
        self.last_considered_snapshot = None

        # MALADA used two times the minimum imaging convention (MIC) to
        # determine the cutoff radius for the RDF.
        # In the future, only the MIC radius should be used, but to
        # recreate old plots, one may need to use two times the MIC.
        if malada_compatability:
            self.target_calculator.parameters.rdf_parameters["rMax"] = "2mic"

    @property
    def trajectory(self):
        """Trajectory to be analyzed."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, new_trajectory):
        self._trajectory = new_trajectory
        self.uncache_properties()

    def _is_property_cached(self, property_name):
        """
        Check if a property is cached.

        Parameters
        ----------
        property_name : str
            Name of the property to check.

        Returns
        -------
        is_cached : bool
            True if the property is cached, False otherwise.
        """
        return property_name in self.__dict__.keys()

    def uncache_properties(self):
        """Uncache all cached properties of this calculator."""
        if self._is_property_cached("first_snapshot"):
            del self.first_snapshot
        if self._is_property_cached("snapshot_correlation_cutoff"):
            del self.snapshot_correlation_cutoff

    @cached_property
    def first_snapshot(self):
        """First equilibrated snapshot."""
        return self.get_first_snapshot()

    @cached_property
    def snapshot_correlation_cutoff(self):
        """Cutoff for the snapshot correlation analysis."""
        return self.get_snapshot_correlation_cutoff()

    def get_first_snapshot(
        self, equilibrated_snapshot=None, distance_threshold=None
    ):
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
        self._distance_metrics = []
        if equilibrated_snapshot is None:
            equilibrated_snapshot = self.trajectory[-1]
        for idx, step in enumerate(self.trajectory):
            self._distance_metrics.append(
                self._calculate_distance_between_snapshots(
                    equilibrated_snapshot,
                    step,
                    "rdf",
                    "cosine_distance",
                    save_rdf1=True,
                )
            )

        # Now, we denoise the distance metrics.
        self.distance_metrics_denoised = self.__denoise(self._distance_metrics)

        # Which snapshots are considered depends on how we denoise the
        # distance metrics.
        self.first_considered_snapshot = (
            self.parameters.trajectory_analysis_denoising_width
        )
        self.last_considered_snapshot = (
            np.shape(self.distance_metrics_denoised)[0]
            - self.parameters.trajectory_analysis_denoising_width
        )
        considered_length = (
            self.last_considered_snapshot - self.first_considered_snapshot
        )

        # Next, the average of the presumed equilibrated part is calculated,
        # and then the first N number of times teps which are below this
        # average is calculated.
        self.average_distance_equilibrated = distance_threshold
        if self.average_distance_equilibrated is None:
            self.average_distance_equilibrated = np.mean(
                self.distance_metrics_denoised[
                    considered_length
                    - int(
                        self.parameters.trajectory_analysis_estimated_equilibrium
                        * considered_length
                    ) : self.last_considered_snapshot
                ]
            )
        is_below = True
        counter = 0
        first_snapshot = None
        for idx, dist in enumerate(self.distance_metrics_denoised):
            if (
                self.first_considered_snapshot
                <= idx
                <= self.last_considered_snapshot
            ):
                if is_below:
                    counter += 1
                if dist < self.average_distance_equilibrated:
                    is_below = True
                if dist >= self.average_distance_equilibrated:
                    counter = 0
                    is_below = False
                if (
                    counter
                    == self.parameters.trajectory_analysis_below_average_counter
                ):
                    first_snapshot = idx
                    break

        printout(
            "First equilibrated timestep of trajectory is", first_snapshot
        )
        return first_snapshot

    def get_snapshot_correlation_cutoff(self):
        """
        Determine the cutoff for the distance metric.

        If a cutoff is set by the user via the Parameters object, this
        function simply returns this value. Elsewise, the value
        is determined numerically. The distance metric used here is realspace
        (i.e. the smallest displacement of an atom between two snapshots). The
        cutoff gives a lower estimate for the oscillations of the trajectory.
        Any distance above this cutoff can be attributed to the oscillations
        in the trajectory. Any cutoff below is the consquence of temporal
        neighborhood of these snapshots.

        Returns
        -------
        cutoff : float
            Cutoff below which two snapshots can be assumed to be similar
            to each other to a degree that suggests temporal neighborhood.

        """
        if self.parameters.trajectory_analysis_correlation_metric_cutoff < 0:
            return self._analyze_distance_metric(self.trajectory)
        else:
            return (
                self.parameters.trajectory_analysis_correlation_metric_cutoff
            )

    def get_uncorrelated_snapshots(self, filename_uncorrelated_snapshots):
        """
        Calculate a set of uncorrelated configurations from a trajectory.

        If not priorly determined with the object calling this function, the
        first equilibrated configuration will be determined automatically.
        This function will create two files, one archiving the MD step number
        of all snapshots sampled, and one with the actual snapshots themselves.

        Parameters
        ----------
        filename_uncorrelated_snapshots : string
            Name of the file in which to save the uncorrelated snapshots.
        """
        filename_base = os.path.basename(
            filename_uncorrelated_snapshots
        ).split(".")[0]
        allowed_temp_diff_K = (
            self.parameters.trajectory_analysis_temperature_tolerance_percent
            / 100
        ) * self.target_calculator.temperature
        current_snapshot = self.first_snapshot
        begin_snapshot = self.first_snapshot + 1
        end_snapshot = len(self.trajectory)
        j = 0
        md_iteration = []
        for i in range(begin_snapshot, end_snapshot):
            if self.__check_if_snapshot_is_valid(
                self.trajectory[i],
                self._temperatures[i],
                self.trajectory[current_snapshot],
                self._temperatures[current_snapshot],
                self.snapshot_correlation_cutoff,
                allowed_temp_diff_K,
            ):
                current_snapshot = i
                md_iteration.append(current_snapshot)
                j += 1
        np.random.shuffle(md_iteration)
        for i in range(0, len(md_iteration)):
            if i == 0:
                traj_writer = TrajectoryWriter(
                    filename_base + ".traj", mode="w"
                )
            else:
                traj_writer = TrajectoryWriter(
                    filename_base + ".traj", mode="a"
                )
            atoms_to_write = Descriptor.enforce_pbc(
                self.trajectory[md_iteration[i]]
            )
            traj_writer.write(atoms=atoms_to_write)
        np.save(filename_base + "_numbers.npy", md_iteration)
        printout(j, "possible snapshots found in MD trajectory.")

    def _analyze_distance_metric(self, trajectory):
        """
        Find threshold for realspace distance trajectory analysis.

        To do that, first the center of the equilibrated part of the trajectory
        is determined, and then the minimum Euclidean distances between this
        snapshot and all snapshots in an appropriate window around it are
        calculated. Based on mean and standard deviation of these distances,
        the threshold is determined.

        Parameters
        ----------
        trajectory : ase.io.Trajectory
            Trajectory to be analyzed.

        Returns
        -------
        threshold : float
            Threshold for the distance (realspace minimum Euclidean distance).
        """
        center = (
            int(
                (
                    np.shape(self.distance_metrics_denoised)[0]
                    - self.first_snapshot
                )
                / 2
            )
            + self.first_snapshot
        )
        width = int(
            self.parameters.trajectory_analysis_estimated_equilibrium
            * np.shape(self.distance_metrics_denoised)[0]
        )
        self.distances_realspace = []
        self.__saved_rdf = None
        for i in range(center - width, center + width):
            self.distances_realspace.append(
                self._calculate_distance_between_snapshots(
                    trajectory[center],
                    trajectory[i],
                    "realspace",
                    "minimal_distance",
                    save_rdf1=True,
                )
            )

        # From these metrics, we assume mean - 2.576 std as limit.
        # This translates to a confidence interval of ~99%, which should
        # make any coincidental similarites unlikely.
        cutoff = np.mean(self.distances_realspace) - 2.576 * np.std(
            self.distances_realspace
        )
        printout("Distance metric cutoff is", cutoff)
        return cutoff

    def _calculate_distance_between_snapshots(
        self,
        snapshot1,
        snapshot2,
        distance_metric,
        reduction,
        save_rdf1=False,
    ):
        """
        Calculate the distance between two snapshots.

        Different metrics and reductions are possible.

        Parameters
        ----------
        snapshot1 : ase.Atoms
            First snapshot to be compared.

        snapshot2 : ase.Atoms
            Second snapshot to be compared.

        distance_metric : string
            Distance metric to be used. Can be "realspace" or "rdf".
            This metric determines _what_ is compared, i.e., atomic positions
            directly or the RDF as representation of them.

        reduction : string
            Reduction to be used. Can be "minimal_distance" or
            "cosine_distance". This metric determines _how_ the representation
            of snapshots are compared.

        save_rdf1 : bool
            If True, the RDF of the first snapshot will be saved for later
            use. This is useful when multiple snapshots are compared to the
            same reference snapshot.

        Returns
        -------
        result : float
            Distance between the two snapshots
        """
        if distance_metric == "realspace":
            positions1 = snapshot1.get_positions()
            positions2 = snapshot2.get_positions()
            if reduction == "minimal_distance":
                result = np.amin(
                    distance.cdist(positions1, positions2), axis=0
                )
                result = np.mean(result)

            elif reduction == "cosine_distance":
                number_of_atoms = snapshot1.get_number_of_atoms()
                result = distance.cosine(
                    np.reshape(positions1, [number_of_atoms * 3]),
                    np.reshape(positions2, [number_of_atoms * 3]),
                )

            else:
                raise Exception("Unknown distance metric reduction.")
        elif distance_metric == "rdf":
            if save_rdf1 is True:
                if self.__saved_rdf is None:
                    self.__saved_rdf = self.target_calculator.get_radial_distribution_function(
                        snapshot1, method="asap3"
                    )[
                        0
                    ]
                rdf1 = self.__saved_rdf
            else:
                rdf1 = self.target_calculator.get_radial_distribution_function(
                    snapshot1, method="asap3"
                )[0]
            rdf2 = self.target_calculator.get_radial_distribution_function(
                snapshot2, method="asap3"
            )[0]

            if reduction == "minimal_distance":
                raise Exception(
                    "Combination of distance metric and reduction "
                    "not supported."
                )

            elif reduction == "cosine_distance":
                result = distance.cosine(rdf1, rdf2)

            else:
                raise Exception("Unknown distance metric reduction.")
        else:
            raise Exception("Unknown distance metric selected.")

        return result

    def __denoise(self, signal):
        """
        Denoise a signal.

        Denoising is performed by a simple moving average.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to be denoised.

        Returns
        -------
        denoised_signal : numpy.ndarray
            Denoised signal.
        """
        denoised_signal = np.convolve(
            signal,
            np.ones(self.parameters.trajectory_analysis_denoising_width)
            / self.parameters.trajectory_analysis_denoising_width,
            mode="same",
        )
        return denoised_signal

    def __check_if_snapshot_is_valid(
        self,
        snapshot_to_test,
        temp_to_test,
        reference_snapshot,
        reference_temp,
        distance_metric,
        allowed_temp_diff,
    ):
        """
        Check if a snapshot is valid to be sampled.

        To be valid, the distance to the last sampled snapshot needs to be
        over a predetermined threshold AND the temperature needs to be
        within a certain window around the target temperature.

        Parameters
        ----------
        snapshot_to_test : ase.Atoms
            Snapshot to check for validity.

        temp_to_test : float
            Temperature of the snapshot to check.

        reference_snapshot : ase.Atoms
            Reference snapshot to compare to (last sampled snapshot).

        reference_temp : float
            Temperature of the reference snapshot (last sampled snapshot).

        distance_metric : float
            Threshold for the distance between the two snapshots.

        allowed_temp_diff : float
            Allowed temperature difference between the two snapshots.

        Returns
        -------
        is_valid : bool
            True if the snapshot is valid, False otherwise.
        """
        distance = self._calculate_distance_between_snapshots(
            snapshot_to_test,
            reference_snapshot,
            "realspace",
            "minimal_distance",
        )
        temp_diff = np.abs(temp_to_test - reference_temp)
        if distance > distance_metric and temp_diff < allowed_temp_diff:
            return True
        else:
            return False
