"""Class for performing a full ACSD analysis."""
import itertools
import os

import numpy as np

from mala.datahandling.data_converter import descriptor_input_types, \
    target_input_types
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target
from mala.network.hyperparameter import Hyperparameter
from mala.network.hyper_opt import HyperOpt
from mala.common.parallelizer import get_rank, printout
from mala.descriptors.bispectrum import Bispectrum
from mala.descriptors.atomic_density import AtomicDensity
from mala.descriptors.minterpy_descriptors import MinterpyDescriptors

descriptor_input_types_acsd = descriptor_input_types+["numpy", "openpmd"]
target_input_types_acsd = target_input_types+["numpy", "openpmd"]


class ACSDAnalyzer(HyperOpt):
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

    def __init__(self, params, target_calculator=None,
                 descriptor_calculator=None):
        super(ACSDAnalyzer, self).__init__(params)
        # Calculators used to parse data from compatible files.
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(params)
        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptor(params)
        if not isinstance(self.descriptor_calculator, Bispectrum) and \
           not isinstance(self.descriptor_calculator, AtomicDensity) and \
           not isinstance(self.descriptor_calculator, MinterpyDescriptors):
            raise Exception("Cannot calculate ACSD for the selected "
                            "descriptors.")

        # Internal variables.
        self.__snapshots = []
        self.__snapshot_description = []
        self.__snapshot_units = []

        # Filled after the analysis.
        self.labels = []
        self.study = []
        self.reduced_study = None
        self.internal_hyperparam_list = None

    def add_snapshot(self, descriptor_input_type=None,
                     descriptor_input_path=None,
                     target_input_type=None,
                     target_input_path=None,
                     descriptor_units=None,
                     target_units=None):
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
                    "Cannot process descriptor data with no path "
                    "given.")
            if descriptor_input_type not in descriptor_input_types_acsd:
                raise Exception(
                    "Cannot process this type of descriptor data.")
        else:
            raise Exception("Cannot calculate ACSD without descriptor data.")

        if target_input_type is not None:
            if target_input_path is None:
                raise Exception("Cannot process target data with no path "
                                "given.")
            if target_input_type not in target_input_types_acsd:
                raise Exception("Cannot process this type of target data.")
        else:
            raise Exception("Cannot calculate ACSD without target data.")

        # Assign info.
        self.__snapshots.append({"input": descriptor_input_path,
                                            "output": target_input_path})
        self.__snapshot_description.append({"input": descriptor_input_type,
                                            "output": target_input_type})
        self.__snapshot_units.append({"input": descriptor_units,
                                      "output": target_units})

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
        if name not in ["bispectrum_twojmax", "bispectrum_cutoff",
                        "atomic_density_sigma", "atomic_density_cutoff",
                        "minterpy_cutoff_cube_size",
                        "minterpy_polynomial_degree",
                        "minterpy_lp_norm"]:
            raise Exception("Unkown hyperparameter for ACSD analysis entered.")

        self.params.hyperparameters.\
            hlist.append(Hyperparameter(hotype="acsd",
                                        name=name,
                                        choices=choices,
                                        opttype="categorical"))

    def perform_study(self, file_based_communication=False,
                      return_plotting=False):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling different descriptors, calculated with
        different hyperparameters and then calculating the ACSD.
        """
        # Prepare the hyperparameter lists.
        self._construct_hyperparam_list()
        hyperparameter_tuples = list(itertools.product(
            *self.internal_hyperparam_list))

        # Perform the ACSD analysis separately for each snapshot.
        best_acsd = None
        best_trial = None
        for i in range(0, len(self.__snapshots)):
            printout("Starting ACSD analysis of snapshot", str(i),
                     min_verbosity=1)
            current_list = []
            target = self._load_target(self.__snapshots[i],
                                       self.__snapshot_description[i],
                                       self.__snapshot_units[i],
                                       file_based_communication)

            for idx, hyperparameter_tuple in enumerate(hyperparameter_tuples):
                if isinstance(self.descriptor_calculator, Bispectrum):
                    self.params.descriptors.bispectrum_cutoff = \
                        hyperparameter_tuple[0]
                    self.params.descriptors.bispectrum_twojmax = \
                        hyperparameter_tuple[1]
                elif isinstance(self.descriptor_calculator, AtomicDensity):
                    self.params.descriptors.atomic_density_cutoff = \
                        hyperparameter_tuple[0]
                    self.params.descriptors.atomic_density_sigma = \
                        hyperparameter_tuple[1]
                elif isinstance(self.descriptor_calculator,
                              MinterpyDescriptors):
                    self.params.descriptors. \
                        atomic_density_cutoff = hyperparameter_tuple[0]
                    self.params.descriptors. \
                        atomic_density_sigma = hyperparameter_tuple[1]
                    self.params.descriptors. \
                        minterpy_cutoff_cube_size = \
                        hyperparameter_tuple[2]
                    self.params.descriptors. \
                        minterpy_polynomial_degree = \
                        hyperparameter_tuple[3]
                    self.params.descriptors. \
                        minterpy_lp_norm = \
                        hyperparameter_tuple[4]

                descriptor = \
                    self._calculate_descriptors(self.__snapshots[i],
                                                self.__snapshot_description[i],
                                                self.__snapshot_units[i])
                if get_rank() == 0:
                    acsd = self._calculate_acsd(descriptor, target,
                                                self.params.hyperparameters.acsd_points,
                                                descriptor_vectors_contain_xyz=
                                                self.params.descriptors.descriptors_contain_xyz)
                    if not np.isnan(acsd):
                        if best_acsd is None:
                            best_acsd = acsd
                            best_trial = idx
                        elif acsd < best_acsd:
                            best_acsd = acsd
                            best_trial = idx
                        current_list.append(list(hyperparameter_tuple) + [acsd])
                    else:
                        current_list.append(list(hyperparameter_tuple) + [np.inf])

                    outstring = "["
                    for label_id, label in enumerate(self.labels):
                        outstring += label + ": " + \
                                     str(hyperparameter_tuple[label_id])
                        if label_id < len(self.labels) - 1:
                            outstring += ", "
                    outstring += "]"
                    best_trial_string = ". No suitable trial found yet."
                    if best_acsd is not None:
                        best_trial_string = ". Best trial is"+str(best_trial) \
                                            + "with"+str(best_acsd)

                    printout("Trial", idx, "finished with ACSD="+str(acsd),
                             "and parameters:", outstring+best_trial_string,
                             min_verbosity=1)

            if get_rank() == 0:
                self.study.append(current_list)

        if get_rank() == 0:
            self.study = np.mean(self.study, axis=0)

            # TODO: Does this even make sense for the minterpy descriptors?
            if return_plotting:
                results_to_plot = []
                if len(self.internal_hyperparam_list) == 2:
                    len_first_dim = len(self.internal_hyperparam_list[0])
                    len_second_dim = len(self.internal_hyperparam_list[1])
                    for i in range(0, len_first_dim):
                        results_to_plot.append(
                            self.study[i*len_second_dim:(i+1)*len_second_dim, 2:])

                    if isinstance(self.descriptor_calculator, Bispectrum):
                        return results_to_plot, {"twojmax": self.internal_hyperparam_list[1],
                                                 "cutoff": self.internal_hyperparam_list[0]}
                    if isinstance(self.descriptor_calculator, AtomicDensity):
                        return results_to_plot, {"sigma": self.internal_hyperparam_list[1],
                                                 "cutoff": self.internal_hyperparam_list[0]}

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        if get_rank() == 0:
            minimum_acsd = self.study[np.argmin(self.study[:, -1])]
            if len(self.internal_hyperparam_list) == 2:
                if isinstance(self.descriptor_calculator, Bispectrum):
                    self.params.descriptors.bispectrum_cutoff = minimum_acsd[0]
                    self.params.descriptors.bispectrum_twojmax = int(minimum_acsd[1])
                    printout("ACSD analysis finished, optimal parameters: ", )
                    printout("Bispectrum twojmax: ", self.params.descriptors.
                             bispectrum_twojmax)
                    printout("Bispectrum cutoff: ", self.params.descriptors.
                             bispectrum_cutoff)
                if isinstance(self.descriptor_calculator, AtomicDensity):
                    self.params.descriptors.atomic_density_cutoff = minimum_acsd[0]
                    self.params.descriptors.atomic_density_sigma = minimum_acsd[1]
                    printout("ACSD analysis finished, optimal parameters: ", )
                    printout("Atomic density sigma: ", self.params.descriptors.
                             atomic_density_sigma)
                    printout("Atomic density cutoff: ", self.params.descriptors.
                             atomic_density_cutoff)
            elif len(self.internal_hyperparam_list) == 5:
                if isinstance(self.descriptor_calculator, MinterpyDescriptors):
                    self.params.descriptors.atomic_density_cutoff = minimum_acsd[0]
                    self.params.descriptors.atomic_density_sigma = minimum_acsd[1]
                    self.params.descriptors.minterpy_cutoff_cube_size = minimum_acsd[2]
                    self.params.descriptors.minterpy_polynomial_degree = int(minimum_acsd[3])
                    self.params.descriptors.minterpy_lp_norm = int(minimum_acsd[4])
                    printout("ACSD analysis finished, optimal parameters: ", )
                    printout("Atomic density sigma: ", self.params.descriptors.
                             atomic_density_sigma)
                    printout("Atomic density cutoff: ", self.params.descriptors.
                             atomic_density_cutoff)
                    printout("Minterpy cube cutoff: ", self.params.descriptors.
                             minterpy_cutoff_cube_size)
                    printout("Minterpy polynomial degree: ", self.params.descriptors.
                             minterpy_polynomial_degree)
                    printout("Minterpy LP norm degree: ", self.params.descriptors.
                             minterpy_lp_norm)

    def _construct_hyperparam_list(self):
        if isinstance(self.descriptor_calculator, Bispectrum):
            if list(map(lambda p: "bispectrum_cutoff" in p.name,
                         self.params.hyperparameters.hlist)).count(True) == 0:
                first_dim_list = [self.params.descriptors.bispectrum_cutoff]
            else:
                first_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "bispectrum_cutoff" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "bispectrum_twojmax" in p.name,
                         self.params.hyperparameters.hlist)).count(True) == 0:
                second_dim_list = [self.params.descriptors.bispectrum_twojmax]
            else:
                second_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "bispectrum_twojmax" in p.name,
                         self.params.hyperparameters.hlist)).index(True)].choices

            self.internal_hyperparam_list = [first_dim_list, second_dim_list]
            self.labels = ["cutoff", "twojmax"]

        elif isinstance(self.descriptor_calculator, AtomicDensity):
            if list(map(lambda p: "atomic_density_cutoff" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                first_dim_list = [self.params.descriptors.atomic_density_cutoff]
            else:
                first_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "atomic_density_cutoff" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "atomic_density_sigma" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                second_dim_list = [self.params.descriptors.atomic_density_sigma]
            else:
                second_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "atomic_density_sigma" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices
            self.internal_hyperparam_list = [first_dim_list, second_dim_list]
            self.labels = ["cutoff", "sigma"]

        elif isinstance(self.descriptor_calculator, MinterpyDescriptors):
            if list(map(lambda p: "atomic_density_cutoff" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                first_dim_list = [self.params.descriptors.atomic_density_cutoff]
            else:
                first_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "atomic_density_cutoff" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "atomic_density_sigma" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                second_dim_list = [self.params.descriptors.atomic_density_sigma]
            else:
                second_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "atomic_density_sigma" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "minterpy_cutoff_cube_size" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                third_dim_list = [self.params.descriptors.minterpy_cutoff_cube_size]
            else:
                third_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "minterpy_cutoff_cube_size" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "minterpy_polynomial_degree" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                fourth_dim_list = [self.params.descriptors.minterpy_polynomial_degree]
            else:
                fourth_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "minterpy_polynomial_degree" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            if list(map(lambda p: "minterpy_lp_norm" in p.name,
                        self.params.hyperparameters.hlist)).count(True) == 0:
                fifth_dim_list = [self.params.descriptors.minterpy_lp_norm]
            else:
                fifth_dim_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "minterpy_lp_norm" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            self.internal_hyperparam_list = [first_dim_list, second_dim_list,
                                             third_dim_list, fourth_dim_list,
                                             fifth_dim_list]
            self.labels = ["cutoff", "sigma", "minterpy_cutoff",
                           "minterpy_polynomial_degree", "minterpy_lp_norm"]

        else:
            raise Exception("Unkown descriptor calculator selected. Cannot "
                            "calculate ACSD.")

    def _calculate_descriptors(self, snapshot, description, original_units):
        descriptor_calculation_kwargs = {}
        tmp_input = None
        if description["input"] == "espresso-out":
            descriptor_calculation_kwargs["units"] = original_units["input"]
            tmp_input, local_size = self.descriptor_calculator. \
                calculate_from_qe_out(snapshot["input"],
                                      **descriptor_calculation_kwargs)

        elif description["input"] is None:
            # In this case, only the output is processed.
            pass

        else:
            raise Exception("Unknown file extension, cannot convert "
                            "descriptor")
        if self.params.descriptors._configuration["mpi"]:
            tmp_input = self.descriptor_calculator. \
                gather_descriptors(tmp_input)

        return tmp_input

    def _load_target(self, snapshot, description, original_units,
                     file_based_communication):
        memmap = None
        if self.params.descriptors._configuration["mpi"] and \
                file_based_communication:
            memmap = "acsd.out.npy_temp"

        target_calculator_kwargs = {}

        # Read the output data
        tmp_output = None
        if description["output"] == ".cube":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = memmap
            # If no units are provided we just assume standard units.
            tmp_output = self.target_calculator. \
                read_from_cube(snapshot["output"],
            ** target_calculator_kwargs)

        elif description["output"] == ".xsf":
            target_calculator_kwargs["units"] = original_units["output"]
            target_calculator_kwargs["use_memmap"] = memmap
            # If no units are provided we just assume standard units.
            tmp_output = self.target_calculator. \
                read_from_xsf(snapshot["output"],
            ** target_calculator_kwargs)

        elif description["output"] == "numpy":
            if get_rank() == 0:
                tmp_output = self.\
                    target_calculator.read_from_numpy_file(
                    snapshot["output"], units=original_units["output"])

        elif description["output"] == "openpmd":
            if get_rank() == 0:
                tmp_output = self.\
                    target_calculator.read_from_numpy_file(
                    snapshot["output"], units=original_units["output"])
        else:
            raise Exception("Unknown file extension, cannot convert target")

        if get_rank() == 0:
            if self.params.targets._configuration["mpi"] \
                    and file_based_communication:
                os.remove(memmap)

        return tmp_output


    @staticmethod
    def _calculate_cosine_similarities(descriptor_data, ldos_data, nr_points,
                                       descriptor_vectors_contain_xyz=True):
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
            descriptor_data = np.reshape(descriptor_data,
                                         (descriptor_dim[0] *
                                          descriptor_dim[1] *
                                          descriptor_dim[2],
                                          descriptor_dim[3]))
            if descriptor_vectors_contain_xyz:
                descriptor_data = descriptor_data[:, 3:]
        elif len(descriptor_dim) != 2:
            raise Exception("Cannot work with this descriptor data.")

        if len(ldos_dim) == 4:
            ldos_data = np.reshape(ldos_data, (ldos_dim[0] * ldos_dim[1] *
                                               ldos_dim[2], ldos_dim[3]))
        elif len(ldos_dim) != 2:
            raise Exception("Cannot work with this LDOS data.")

        similarity_array = []
        # Draw nr_points at random from snapshot.
        rng = np.random.default_rng()
        points_i = rng.choice(np.shape(descriptor_data)[0],
                              size=np.shape(descriptor_data)[0],
                              replace=False)
        for i in range(0, nr_points):
            # Draw another nr_points at random from snapshot.
            rng = np.random.default_rng()
            points_j = rng.choice(np.shape(descriptor_data)[0],
                                  size=np.shape(descriptor_data)[0],
                                  replace=False)

            for j in range(0, nr_points):
                # Calculate similarities between these two pairs.
                descriptor_distance = \
                    ACSDAnalyzer.__calc_cosine_similarity(
                        descriptor_data[points_i[i]],
                        descriptor_data[points_j[j]])
                ldos_distance = ACSDAnalyzer.\
                    __calc_cosine_similarity(ldos_data[points_i[i]],
                                             ldos_data[points_j[j]])
                similarity_array.append([descriptor_distance, ldos_distance])

        return np.array(similarity_array)

    @staticmethod
    def _calculate_acsd(descriptor_data, ldos_data, acsd_points,
                        descriptor_vectors_contain_xyz=True):
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
        def distance_between_points(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        similarity_data = ACSDAnalyzer.\
            _calculate_cosine_similarities(descriptor_data, ldos_data,
                                           acsd_points,
                                           descriptor_vectors_contain_xyz=
                                           descriptor_vectors_contain_xyz)
        data_size = np.shape(similarity_data)[0]
        distances = []
        for i in range(0, data_size):
            distances.append(distance_between_points(similarity_data[i, 0],
                                                     similarity_data[i, 1],
                                                     similarity_data[i, 0],
                                                     similarity_data[i, 0]))
        return np.mean(distances)

    @staticmethod
    def __calc_cosine_similarity(vector1, vector2, norm=2):
        if np.shape(vector1)[0] != np.shape(vector2)[0]:
            raise Exception("Cannot calculate similarity between vectors "
                            "of different dimenstions.")
        if np.shape(vector1)[0] == 1:
            return np.min([vector1[0], vector2[0]]) / \
                   np.max([vector1[0], vector2[0]])
        else:
            return np.dot(vector1, vector2) / \
                   (np.linalg.norm(vector1, ord=norm) *
                    np.linalg.norm(vector2, ord=norm))
