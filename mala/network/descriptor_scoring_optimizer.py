"""Base class for ACSD, mutual information and related methods."""

from abc import abstractmethod, ABC
import itertools
import os

import numpy as np

from mala.datahandling.data_converter import (
    descriptor_input_types,
    target_input_types,
)
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target
from mala.network.hyperparameter import Hyperparameter
from mala.network.hyper_opt import HyperOpt
from mala.common.parallelizer import get_rank, printout
from mala.descriptors.bispectrum import Bispectrum
from mala.descriptors.atomic_density import AtomicDensity
from mala.descriptors.minterpy_descriptors import MinterpyDescriptors

descriptor_input_types_descriptor_scoring = descriptor_input_types + [
    "numpy",
    "openpmd",
    "json",
]
target_input_types_descriptor_scoring = target_input_types + [
    "numpy",
    "openpmd",
]


class DescriptorScoringOptimizer(HyperOpt, ABC):
    """
    Base class for all training-free descriptor hyperparameter optimizers.

    These optimizer use alternative metrics ACSD, mutual information, etc.
    to tune descriptor hyperparameters.

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

    Attributes
    ----------
    best_score : float
        Score associated with best-performing trial.

    best_trial_index : int
        Index of best-performing trial

    labels : list
        Contains the labels of all the hyperparameters sampled during analysis.

    averaged_study : list
        Contains results of the analysis averaged over all snapshots.
        Can be used for plotting.
    """

    def __init__(
        self, params, target_calculator=None, descriptor_calculator=None
    ):
        super(DescriptorScoringOptimizer, self).__init__(params)
        # Calculators used to parse data from compatible files.
        self._target_calculator = target_calculator
        if self._target_calculator is None:
            self._target_calculator = Target(params)
        self._descriptor_calculator = descriptor_calculator
        if self._descriptor_calculator is None:
            self._descriptor_calculator = Descriptor(params)
        if not isinstance(
            self._descriptor_calculator, Bispectrum
        ) and not isinstance(self._descriptor_calculator, AtomicDensity):
            raise Exception("Unsupported descriptor type selected.")

        # Internal variables.
        self._snapshots = []
        self._snapshot_description = []
        self._snapshot_units = []

        # Filled after the analysis.
        self._study = []
        self._internal_hyperparam_list = None

        # Logging metrics.
        self.best_score = None
        self.best_trial_index = None

        # For plotting.
        self.labels = []
        self.averaged_study = None

    def add_snapshot(
        self,
        descriptor_input_type=None,
        descriptor_input_path=None,
        target_input_type=None,
        target_input_path=None,
        descriptor_units=None,
        target_units=None,
    ):
        """
        Add a snapshot to be processed.

        Parameters
        ----------
        descriptor_input_type : string
            Type of descriptor data to be processed.
            See mala.datahandling.data_converter.descriptor_input_types
            for options.

        descriptor_input_path : string
            Path of descriptor data to be processed.

        target_input_type : string
            Type of target data to be processed.
            See mala.datahandling.data_converter.target_input_types
            for options.

        target_input_path : string
            Path of target data to be processed.

        descriptor_units : string
            Units for descriptor data processing.

        target_units : string
            Units for target data processing.
        """
        # Check the input.
        if descriptor_input_type is not None:
            if descriptor_input_path is None:
                raise Exception(
                    "Cannot process descriptor data with no path given."
                )
            if (
                descriptor_input_type
                not in descriptor_input_types_descriptor_scoring
            ):
                raise Exception("Cannot process this type of descriptor data.")
        else:
            raise Exception(
                "Cannot calculate scoring metrics without descriptor data."
            )

        if target_input_type is not None:
            if target_input_path is None:
                raise Exception(
                    "Cannot process target data with no path given."
                )
            if target_input_type not in target_input_types_descriptor_scoring:
                raise Exception("Cannot process this type of target data.")
        else:
            raise Exception(
                "Cannot calculate scoring metrics without target data."
            )

        # Assign info.
        self._snapshots.append(
            {"input": descriptor_input_path, "output": target_input_path}
        )
        self._snapshot_description.append(
            {"input": descriptor_input_type, "output": target_input_type}
        )
        self._snapshot_units.append(
            {"input": descriptor_units, "output": target_units}
        )

    def add_hyperparameter(self, name, choices):
        """
        Add a hyperparameter to the current investigation.

        Parameters
        ----------
        name : string
            Name of the hyperparameter. Please note that these names always
            have to be the same as the parameter names in
            ParametersDescriptors.

        choices :
            List of possible choices.
        """
        if name not in [
            "bispectrum_twojmax",
            "bispectrum_cutoff",
            "bispectrum_element_weights",
            "atomic_density_sigma",
            "atomic_density_cutoff",
        ]:
            raise Exception(
                "Unkown hyperparameter for training free descriptor"
                "hyperparameter optimization entered."
            )

        self.params.hyperparameters.hlist.append(
            Hyperparameter(
                hotype="descriptor_scoring",
                name=name,
                choices=choices,
                opttype="categorical",
            )
        )

    def perform_study(self, file_based_communication=False):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling different descriptors, calculated with
        different hyperparameters and then calculating some surrogate score.
        """
        # Prepare the hyperparameter lists.
        self._construct_hyperparam_list()
        hyperparameter_tuples = list(
            itertools.product(*self._internal_hyperparam_list)
        )

        # Perform the descriptor scoring analysis separately for each snapshot.
        for i in range(0, len(self._snapshots)):
            self.best_trial_index = None
            self.best_score = None
            printout(
                "Starting descriptor scoring analysis of snapshot",
                str(i),
                min_verbosity=1,
            )
            current_list = []
            target = self._load_target(
                self._snapshots[i],
                self._snapshot_description[i],
                self._snapshot_units[i],
                file_based_communication,
            )

            for idx, hyperparameter_tuple in enumerate(hyperparameter_tuples):
                if isinstance(self._descriptor_calculator, Bispectrum):
                    self.params.descriptors.bispectrum_cutoff = (
                        hyperparameter_tuple[0]
                    )
                    self.params.descriptors.bispectrum_twojmax = (
                        hyperparameter_tuple[1]
                    )
                    self.params.descriptors.bispectrum_element_weights = [
                        float(x) for x in hyperparameter_tuple[2].split(",")
                    ]
                elif isinstance(self._descriptor_calculator, AtomicDensity):
                    self.params.descriptors.atomic_density_cutoff = (
                        hyperparameter_tuple[0]
                    )
                    self.params.descriptors.atomic_density_sigma = (
                        hyperparameter_tuple[1]
                    )

                descriptor = self._calculate_descriptors(
                    self._snapshots[i],
                    self._snapshot_description[i],
                    self._snapshot_units[i],
                )
                if get_rank() == 0:
                    score = self._calculate_score(
                        descriptor,
                        target,
                    )
                    if not np.isnan(score):
                        self._update_logging(score, idx)
                        current_list.append(
                            list(hyperparameter_tuple) + [score]
                        )
                    else:
                        current_list.append(
                            list(hyperparameter_tuple) + [np.inf]
                        )

                    outstring = "["
                    for label_id, label in enumerate(self.labels):
                        outstring += (
                            label + ": " + str(hyperparameter_tuple[label_id])
                        )
                        if label_id < len(self.labels) - 1:
                            outstring += ", "
                    outstring += "]"
                    best_trial_string = ". No suitable trial found yet."
                    if self.best_score is not None:
                        best_trial_string = (
                            ". Best trial is "
                            + str(self.best_trial_index)
                            + " with "
                            + str(self.best_score)
                        )

                    printout(
                        "Trial",
                        idx,
                        "finished with score=" + str(score),
                        "and parameters:",
                        outstring + best_trial_string,
                        min_verbosity=1,
                    )

            if get_rank() == 0:
                self._study.append(current_list)

        if get_rank() == 0:
            averaged_results = np.mean(
                np.float64(np.array(self._study)[:, :, -1]), axis=0
            )

            self.averaged_study = []
            for i in range(0, len(hyperparameter_tuples)):
                self.averaged_study.append(self._study[0][i])
                self.averaged_study[i][-1] = averaged_results[i]

    def set_optimal_parameters(self):
        """
        Set optimal parameters.

        This function will write the determined hyperparameters directly to
        MALA parameters object referenced in this class.
        """
        if get_rank() == 0:
            best_trial = self._get_best_trial()
            if isinstance(self._descriptor_calculator, Bispectrum):
                self.params.descriptors.bispectrum_cutoff = best_trial[0]
                self.params.descriptors.bispectrum_twojmax = int(best_trial[1])
                self.params.descriptors.bispectrum_element_weights = [
                    float(x) for x in best_trial[2].split(",")
                ]
                printout(
                    "Descriptor scoring analysis finished, optimal parameters: ",
                )
                printout(
                    "Bispectrum twojmax: ",
                    self.params.descriptors.bispectrum_twojmax,
                )
                printout(
                    "Bispectrum cutoff: ",
                    self.params.descriptors.bispectrum_cutoff,
                )
                printout(
                    "Bispectrum element weights: ",
                    self.params.descriptors.bispectrum_element_weights,
                )
            if isinstance(self._descriptor_calculator, AtomicDensity):
                self.params.descriptors.atomic_density_cutoff = best_trial[0]
                self.params.descriptors.atomic_density_sigma = best_trial[1]
                printout(
                    "Descriptor scoring analysis finished, optimal parameters: ",
                )
                printout(
                    "Atomic density sigma: ",
                    self.params.descriptors.atomic_density_sigma,
                )
                printout(
                    "Atomic density cutoff: ",
                    self.params.descriptors.atomic_density_cutoff,
                )

    @abstractmethod
    def _get_best_trial(self):
        """Determine the best trial as given by this study."""
        pass

    def _construct_hyperparam_list(self):
        """
        Construct the hyperparameter list for the descriptor scoring analysis.

        Based on a list of choices for descriptor hyperparameters, this
        function creates a list of trials to be tested.
        """

        def hyperparameter_in_hlist(hyperparameter_name):
            return (
                list(
                    map(
                        lambda p: hyperparameter_name in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).count(True)
                > 0
            )

        def extract_choices_from_hlist(hyperparameter_name):
            return self.params.hyperparameters.hlist[
                list(
                    map(
                        lambda p: hyperparameter_name in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).index(True)
            ].choices

        self._internal_hyperparam_list = []

        if isinstance(self._descriptor_calculator, Bispectrum):
            if hyperparameter_in_hlist("bispectrum_cutoff"):
                self._internal_hyperparam_list.append(
                    extract_choices_from_hlist("bispectrum_cutoff")
                )
            else:
                self._internal_hyperparam_list.append(
                    [self.params.descriptors.bispectrum_cutoff]
                )

            if hyperparameter_in_hlist("bispectrum_twojmax"):
                self._internal_hyperparam_list.append(
                    extract_choices_from_hlist("bispectrum_twojmax")
                )
            else:
                self._internal_hyperparam_list.append(
                    [self.params.descriptors.bispectrum_twojmax]
                )

            if hyperparameter_in_hlist("bispectrum_element_weights"):
                self._internal_hyperparam_list.append(
                    extract_choices_from_hlist("bispectrum_element_weights")
                )
            else:
                self._internal_hyperparam_list.append(
                    [self.params.descriptors.bispectrum_element_weights]
                )

            self.labels = ["cutoff", "twojmax", "element_weights"]

        elif isinstance(self._descriptor_calculator, AtomicDensity):
            if hyperparameter_in_hlist("atomic_density_cutoff"):
                self._internal_hyperparam_list.append(
                    extract_choices_from_hlist("atomic_density_cutoff")
                )
            else:
                self._internal_hyperparam_list.append(
                    [self.params.descriptors.atomic_density_cutoff]
                )
            if hyperparameter_in_hlist("atomic_density_sigma"):
                self._internal_hyperparam_list.append(
                    extract_choices_from_hlist("atomic_density_sigma")
                )
            else:
                self._internal_hyperparam_list.append(
                    [self.params.descriptors.atomic_density_sigma]
                )

            self.labels = ["cutoff", "sigma"]

        else:
            raise Exception(
                "Unkown descriptor calculator selected. Cannot "
                "perform descriptor scoring optimization."
            )

    def _calculate_descriptors(self, snapshot, description, original_units):
        """
        Calculate descriptors from a dictionary with snapshot information.

        Internally uses the DescriptorCalculator class.

        Parameters
        ----------
        snapshot : dict
            Dictionary containing information on "input" and "output" files.

        description : dict
            Dictionary containing information on "input" and "output"
            file types.

        original_units : dict
            Dictionary containing information on "input" and "output" units.

        Returns
        -------
        descriptor_data : numpy.ndarray
            Array containing the descriptor data.
        """
        descriptor_calculation_kwargs = {}
        tmp_input = None
        if description["input"] == "espresso-out":
            descriptor_calculation_kwargs["units"] = original_units["input"]
            tmp_input, local_size = (
                self._descriptor_calculator.calculate_from_qe_out(
                    snapshot["input"], **descriptor_calculation_kwargs
                )
            )

        elif description["input"] is None:
            # In this case, only the output is processed.
            pass
        elif description["input"] == "json":
            # In this case, only the output is processed.
            descriptor_calculation_kwargs["units"] = original_units["input"]
            tmp_input, local_size = (
                self._descriptor_calculator.calculate_from_json(
                    snapshot["input"], **descriptor_calculation_kwargs
                )
            )

        else:
            raise Exception(
                "Unknown file extension, cannot convert descriptor"
            )
        if self.params.descriptors._configuration["mpi"]:
            tmp_input = self._descriptor_calculator.gather_descriptors(
                tmp_input
            )

        return tmp_input

    def _load_target(
        self, snapshot, description, original_units, file_based_communication
    ):
        """
        Load target data from a dictionary with snapshot information.

        Internally uses the TargetCalculator class.

        Parameters
        ----------
        snapshot : dict
            Dictionary containing information on "input" and "output" files.

        description : dict
            Dictionary containing information on "input" and "output"
            file types.

        original_units : dict
            Dictionary containing information on "input" and "output" units.

        file_based_communication : bool
            If True, file-based communication is used for MPI. This means that
            a temporary file is used to collect target components across
            ranks rather than MPI communication directly. This can help
            with memory issues when the target data is large.

        Returns
        -------
        target_data : numpy.ndarray
            Array containing the target data.
        """
        memmap = None
        if (
            self.params.descriptors._configuration["mpi"]
            and file_based_communication
        ):
            memmap = "descriptor_scoring.out.npy_temp"

        target_calculator_kwargs = {}

        # Read the output data
        tmp_output = None
        if description["output"] == ".cube":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = memmap
            # If no units are provided we just assume standard units.
            tmp_output = self._target_calculator.read_from_cube(
                snapshot["output"], **target_calculator_kwargs
            )

        elif description["output"] == ".xsf":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = memmap
            # If no units are provided we just assume standard units.
            tmp_output = self._target_calculator.read_from_xsf(
                snapshot["output"], **target_calculator_kwargs
            )

        elif description["output"] == "numpy":
            if get_rank() == 0:
                tmp_output = self._target_calculator.read_from_numpy_file(
                    snapshot["output"], units=original_units["output"]
                )

        elif description["output"] == "openpmd":
            if get_rank() == 0:
                tmp_output = self._target_calculator.read_from_numpy_file(
                    snapshot["output"], units=original_units["output"]
                )
        else:
            raise Exception("Unknown file extension, cannot convert target")

        if get_rank() == 0:
            if (
                self.params.targets._configuration["mpi"]
                and file_based_communication
            ):
                os.remove(memmap)

        return tmp_output

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
