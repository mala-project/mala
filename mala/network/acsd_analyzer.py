"""Class for performing a full ACSD analysis."""
import os

import numpy as np

from mala.datahandling.data_converter import descriptor_input_types, \
    target_input_types
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target
from mala.network.hyperparameter import Hyperparameter
from mala.network.hyper_opt import HyperOpt
from mala.common.parallelizer import get_rank, printout

descriptor_input_types_acsd = descriptor_input_types+["numpy", "openpmd"]
target_input_types_acsd = target_input_types+["numpy", "openpmd"]


class ACSDAnalyzer(HyperOpt):

    def __int__(self, params, target_calculator=None,
                descriptor_calculator=None):
        # Calculators used to parse data from compatible files.
        self.target_calculator = target_calculator
        if self.target_calculator is None:
            self.target_calculator = Target(params)
        self.descriptor_calculator = descriptor_calculator
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptor(params)

        # Internal variables.
        self.__snapshots = []
        self.__snapshot_description = []
        self.__snapshot_units = []

        # Filled after the analysis.
        self.labels = []
        self.study = []

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
        if name != "bispectrum_twojmax" and \
           name != "bispectrum_cutoff" and \
           name != "atomic_density_sigma" and \
           name != "atomic_density_cutoff":
            raise Exception("Unkown hyperparameter for ACSD analysis entered.")

        self.params.hyperparameters.hlist.append(Hyperparameter(name=name,
                                                    choices=choices))

    def perform_study(self, file_based_communication=False):
        # The individual loops depend on the type of descriptors.
        if self.params.descriptors.descriptor_type == "Bispectrum":
            if list(map(lambda p: "bispectrum_twojmax" in p.name,
                         self.params.hyperparameters.hlist)).count(True) == 0:
                twojmax_list = [self.params.descriptors.bispectrum_twojmax]
            else:
                twojmax_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "bispectrum_twojmax" in p.name,
                         self.params.hyperparameters.hlist)).index(True)].choices
            if list(map(lambda p: "bispectrum_cutoff" in p.name,
                         self.params.hyperparameters.hlist)).count(True) == 0:
                cutoff_list = [self.params.descriptors.bispectrum_cutoff]
            else:
                cutoff_list = \
                    self.params.hyperparameters.hlist[
                        list(map(lambda p: "bispectrum_cutoff" in p.name,
                                 self.params.hyperparameters.hlist)).index(
                            True)].choices

            self.labels = ["twojmax", "cutoff", "acsd"]

            # Perform the ACSD analysis separately for each snapshot.
            for i in range(0, len(self.__snapshots)):
                target = self._load_target(self.__snapshots[i],
                                           self.__snapshot_description[i],
                                           self.__snapshot_units[i],
                                           file_based_communication)
                for twojmax in twojmax_list:
                    for cutoff in cutoff_list:
                        self.params.descriptors.bispectrum_twojmax = twojmax
                        self.params.descriptors.bispectrum_cutoff = cutoff
                        descriptor = \
                            self._calculate_descriptors(self.__snapshots[i],
                                                        self.__snapshot_description[i],
                                                        self.__snapshot_units[i])
                        if get_rank() == 0:
                            acsd = self._calculate_acsd(descriptor, target,
                                                 self.params.hyperparameters.acsd_points,
                                                 descriptor_vectors_contain_xyz=
                                                 self.params.descriptors.descriptors_contain_xyz)
                            self.study.append([twojmax, cutoff, acsd])

    def set_optimal_parameters(self):
        minimum_acsd = self.study[np.argmin(self.study[:, 2])]
        if self.params.descriptors.descriptor_type == "Bispectrum":
            self.params.descriptors.bispectrum_twojmax = minimum_acsd[0]
            self.params.descriptors.bispectrum_cutoff = minimum_acsd[1]
            printout("Optimal parameters: ", )
            printout("Bispectrum twojmax: ", self.params.descriptors.
                     bispectrum_twojmax)
            printout("Bispectrum cutoff: ", self.params.descriptors.
                     bispectrum_cutoff)

    def _calculate_descriptors(self, snapshot, description, original_units):
        descriptor_calculation_kwargs = {}
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
        tmp_input = None
        if self.params.descriptors._configuration["mpi"]:
            tmp_input = self.descriptor_calculator. \
                gather_descriptors(tmp_input)

        return tmp_input

    def _load_target(self, snapshot, description, original_units,
                     file_based_communication):
        memmap = None
        if self.params._configuration["mpi"] and \
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
            if self.params._configuration["mpi"] \
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
