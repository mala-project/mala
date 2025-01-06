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
        self._labels = []
        self._study = []
        self._reduced_study = None
        self._internal_hyperparam_list = None

        # Logging metrics.
        self.best_score = None
        self.best_trial_index = None

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

    def perform_study(
        self, file_based_communication=False, return_plotting=False
    ):
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
                    for label_id, label in enumerate(self._labels):
                        outstring += (
                            label + ": " + str(hyperparameter_tuple[label_id])
                        )
                        if label_id < len(self._labels) - 1:
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
            self._study = np.mean(self._study, axis=0)

            if return_plotting:
                results_to_plot = []
                if len(self._internal_hyperparam_list) == 2:
                    len_first_dim = len(self._internal_hyperparam_list[0])
                    len_second_dim = len(self._internal_hyperparam_list[1])
                    for i in range(0, len_first_dim):
                        results_to_plot.append(
                            self._study[
                                i * len_second_dim : (i + 1) * len_second_dim,
                                2:,
                            ]
                        )

                    if isinstance(self._descriptor_calculator, Bispectrum):
                        return results_to_plot, {
                            "twojmax": self._internal_hyperparam_list[1],
                            "cutoff": self._internal_hyperparam_list[0],
                        }
                    if isinstance(self._descriptor_calculator, AtomicDensity):
                        return results_to_plot, {
                            "sigma": self._internal_hyperparam_list[1],
                            "cutoff": self._internal_hyperparam_list[0],
                        }

    def set_optimal_parameters(self):
        """
        Set optimal parameters.

        This function will write the determined hyperparameters directly to
        MALA parameters object referenced in this class.
        """
        if get_rank() == 0:
            best_trial = self._get_best_trial()
            minimum_score = self._study[np.argmin(self._study[:, -1])]
            if isinstance(self._descriptor_calculator, Bispectrum):
                self.params.descriptors.bispectrum_cutoff = best_trial[0]
                self.params.descriptors.bispectrum_twojmax = int(best_trial[1])
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
        if isinstance(self._descriptor_calculator, Bispectrum):
            if (
                list(
                    map(
                        lambda p: "bispectrum_cutoff" in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).count(True)
                == 0
            ):
                first_dim_list = [self.params.descriptors.bispectrum_cutoff]
            else:
                first_dim_list = self.params.hyperparameters.hlist[
                    list(
                        map(
                            lambda p: "bispectrum_cutoff" in p.name,
                            self.params.hyperparameters.hlist,
                        )
                    ).index(True)
                ].choices

            if (
                list(
                    map(
                        lambda p: "bispectrum_twojmax" in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).count(True)
                == 0
            ):
                second_dim_list = [self.params.descriptors.bispectrum_twojmax]
            else:
                second_dim_list = self.params.hyperparameters.hlist[
                    list(
                        map(
                            lambda p: "bispectrum_twojmax" in p.name,
                            self.params.hyperparameters.hlist,
                        )
                    ).index(True)
                ].choices

            self._internal_hyperparam_list = [first_dim_list, second_dim_list]
            self._labels = ["cutoff", "twojmax"]

        elif isinstance(self._descriptor_calculator, AtomicDensity):
            if (
                list(
                    map(
                        lambda p: "atomic_density_cutoff" in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).count(True)
                == 0
            ):
                first_dim_list = [
                    self.params.descriptors.atomic_density_cutoff
                ]
            else:
                first_dim_list = self.params.hyperparameters.hlist[
                    list(
                        map(
                            lambda p: "atomic_density_cutoff" in p.name,
                            self.params.hyperparameters.hlist,
                        )
                    ).index(True)
                ].choices

            if (
                list(
                    map(
                        lambda p: "atomic_density_sigma" in p.name,
                        self.params.hyperparameters.hlist,
                    )
                ).count(True)
                == 0
            ):
                second_dim_list = [
                    self.params.descriptors.atomic_density_sigma
                ]
            else:
                second_dim_list = self.params.hyperparameters.hlist[
                    list(
                        map(
                            lambda p: "atomic_density_sigma" in p.name,
                            self.params.hyperparameters.hlist,
                        )
                    ).index(True)
                ].choices
            self._internal_hyperparam_list = [first_dim_list, second_dim_list]
            self._labels = ["cutoff", "sigma"]

        else:
            raise Exception(
                "Unkown descriptor calculator selected. Cannot "
                "perform descriptor scoring optimization."
            )

    def _calculate_descriptors(self, snapshot, description, original_units):
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
        pass

    @abstractmethod
    def _calculate_score(self, descriptor, target):
        pass
