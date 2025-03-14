"""Base class for all calculators that deal with physical data."""

from abc import ABC, abstractmethod
import os

import json
import numpy as np
from mala.common.parallelizer import get_comm, get_rank

from mala.version import __version__ as mala_version


class PhysicalData(ABC):
    """
    Base class for volumetric physical data.

    Implements general framework to read and write such data to and from
    files. Volumetric data is assumed to exist on a 3D grid. As such it
    either has the dimensions [x,y,z,f], where f is the feature dimension.
    All loading functions within this class assume such a 4D array. Within
    MALA, occasionally 2D arrays of dimension [x*y*z,f] are used and reshaped
    accordingly.

    Parameters
    ----------
    parameters : mala.Parameters
        MALA Parameters object used to create this class.

    Attributes
    ----------
    parameters : mala.Parameters
        MALA parameters object.

    grid_dimensions : list
        List of the grid dimensions (x,y,z)
    """

    ##############################
    # Constructors
    ##############################

    def __init__(self, parameters):
        self.parameters = parameters
        self.grid_dimensions = [0, 0, 0]

    ##############################
    # Properties
    ##############################

    @property
    @abstractmethod
    def data_name(self):
        """Get a string that describes the data (for e.g. metadata)."""
        pass

    @property
    @abstractmethod
    def feature_size(self):
        """Get the feature dimension of this data."""
        pass

    @property
    @abstractmethod
    def si_dimension(self):
        """
        Dictionary containing the SI unit dimensions in OpenPMD format.

        Needed for OpenPMD interface.
        """
        pass

    @property
    @abstractmethod
    def si_unit_conversion(self):
        """
        Numeric value of the conversion from MALA (ASE) units to SI.

        Needed for OpenPMD interface.
        """
        pass

    ##############################
    # Read functions.
    #   Write functions for now not implemented at this level
    #   because there is no need to.
    ##############################

    def read_from_numpy_file(
        self, path, units=None, array=None, reshape=False
    ):
        """
        Read the data from a numpy file.

        Parameters
        ----------
        path : string
            Path to the numpy file.

        units : string
            Units the data is saved in.

        array : np.ndarray
            If not None, the array to save the data into.
            The array has to be 4-dimensional.

        reshape : bool
            If True, the loaded 4D array will be reshaped into a 2D array.

        Returns
        -------
        data : numpy.ndarray or None
            If array is None, a numpy array containing the data.
            Elsewise, None, as the data will be saved into the provided
            array.

        """
        if array is None:
            loaded_array = np.load(path)[:, :, :, self._feature_mask() :]
            self._process_loaded_array(loaded_array, units=units)
            return loaded_array
        else:
            if reshape:
                array_dims = np.shape(array)
                array[:, :] = np.load(path)[
                    :, :, :, self._feature_mask() :
                ].reshape(array_dims)
            else:
                array[:, :, :, :] = np.load(path)[
                    :, :, :, self._feature_mask() :
                ]
            self._process_loaded_array(array, units=units)

    def read_from_openpmd_file(self, path, units=None, array=None):
        """
        Read the data from a numpy file.

        Parameters
        ----------
        path : string
            Path to the openPMD file.

        units : string
            Units the data is saved in.

        array : np.ndarray
            If not None, the array to save the data into.
            The array has to be 4-dimensional.

        Returns
        -------
        data : numpy.ndarray or None
            If array is None, a numpy array containing the data.
            Elsewise, None, as the data will be saved into the provided
            array.
        """
        import openpmd_api as io

        # The union operator for dicts is only supported starting with
        # python 3.9. Currently, MALA works down to python 3.8; For now,
        # I think it is good to keep it that way.
        # I will leave this code in for now, because that may change in the
        # future.
        # series = io.Series(path, io.Access.read_only,
        #                    options=json.dumps(
        #                         {"defer_iteration_parsing": True} |
        #                         self.parameters.
        #                             _configuration["openpmd_configuration"]))
        options = self.parameters._configuration[
            "openpmd_configuration"
        ].copy()
        options["defer_iteration_parsing"] = True
        series = io.Series(
            path, io.Access.read_only, options=json.dumps(options)
        )

        # Check if this actually MALA compatible data.
        if series.get_attribute("is_mala_data") != 1:
            raise Exception(
                "Non-MALA data detected, cannot work with this data."
            )

        # A bit clanky, but this way only the FIRST iteration is loaded,
        # which is what we need for loading from a single file that
        # may be whatever iteration in its series.
        # Also, in combination with `defer_iteration_parsing`, specified as
        # default above, this opens and parses the first iteration,
        # and no others.
        for current_iteration in series.read_iterations():
            mesh = current_iteration.meshes[self.data_name]
            break

        # Read the attributes from the OpenPMD iteration.
        self._process_openpmd_attributes(series, current_iteration, mesh)

        # TODO: Are there instances in MALA, where we wouldn't just label
        # the feature dimension with 0,1,... ? I can't think of one.
        # But there may be in the future, and this'll break
        if array is None:
            data = np.zeros(
                (
                    mesh["0"].shape[0],
                    mesh["0"].shape[1],
                    mesh["0"].shape[2],
                    len(mesh) - self._feature_mask(),
                ),
                dtype=mesh["0"].dtype,
            )
        else:
            if (
                array.shape[0] != mesh["0"].shape[0]
                or array.shape[1] != mesh["0"].shape[1]
                or array.shape[2] != mesh["0"].shape[2]
                or array.shape[3] != len(mesh) - self._feature_mask()
            ):
                raise Exception(
                    "Cannot load data into array, wrong shape provided."
                )

        # Only check this once, since we do not save arrays with different
        # units throughout the feature dimension.
        # Later, we can merge this unit check with the unit conversion
        # MALA does naturally.
        if not np.isclose(mesh[str(0)].unit_SI, self.si_unit_conversion):
            raise Exception(
                "MALA currently cannot operate with OpenPMD "
                "files with non-MALA units."
            )

        # Deal with `granularity` items of the vectors at a time
        # Or in the openPMD layout: with `granularity` record components
        granularity = self.parameters._configuration["openpmd_granularity"]

        if array is None:
            array_shape = data.shape
            data_type = data.dtype
        else:
            array_shape = array.shape
            data_type = array.dtype
        for base in range(
            self._feature_mask(),
            array_shape[3] + self._feature_mask(),
            granularity,
        ):
            end = min(
                base + granularity, array_shape[3] + self._feature_mask()
            )
            transposed = np.empty(
                (end - base, array_shape[0], array_shape[1], array_shape[2]),
                dtype=data_type,
            )
            for i in range(base, end):
                mesh[str(i)].load_chunk(transposed[i - base, :, :, :])
            series.flush()
            if array is None:
                data[
                    :,
                    :,
                    :,
                    base - self._feature_mask() : end - self._feature_mask(),
                ] = np.transpose(transposed, axes=[1, 2, 3, 0])[:, :, :, :]
            else:
                array[
                    :,
                    :,
                    :,
                    base - self._feature_mask() : end - self._feature_mask(),
                ] = np.transpose(transposed, axes=[1, 2, 3, 0])[:, :, :, :]

        if array is None:
            self._process_loaded_array(data, units=units)
            return data
        else:
            self._process_loaded_array(array, units=units)

    def read_dimensions_from_numpy_file(self, path, read_dtype=False):
        """
        Read only the dimensions from a numpy file.

        Parameters
        ----------
        path : string
            Path to the numpy file.

        read_dtype : bool
            If True, the dtype is read alongside the dimensions.

        Returns
        -------
        dimension_info : list or tuple
            If read_dtype is False, then only a list containing the dimensions
            of the saved array is returned. If read_dtype is True, a tuple
            containing this list of dimensions and the dtype of the array will
            be returned.
        """
        loaded_array = np.load(path, mmap_mode="r")
        if read_dtype:
            return (
                self._process_loaded_dimensions(np.shape(loaded_array)),
                loaded_array.dtype,
            )
        else:
            return self._process_loaded_dimensions(np.shape(loaded_array))

    def read_dimensions_from_openpmd_file(
        self, path, comm=None, read_dtype=False
    ):
        """
        Read only the dimensions from a openPMD file.

        Parameters
        ----------
        path : string
            Path to the openPMD file.

        read_dtype : bool
            If True, the dtype is read alongside the dimensions.

        comm : MPI.Comm
            An MPI communicator to be used for parallelized I/O

        Returns
        -------
        dimension_info : list
            A list containing the dimensions of the saved array.
        """
        if comm is None or comm.rank == 0:
            import openpmd_api as io

            # The union operator for dicts is only supported starting with
            # python 3.9. Currently, MALA works down to python 3.8; For now,
            # I think it is good to keep it that way.
            # I will leave this code in for now, because that may change in the
            # future.
            # series = io.Series(path, io.Access.read_only,
            #                    options=json.dumps(
            #                         {"defer_iteration_parsing": True} |
            #                         self.parameters.
            #                             _configuration["openpmd_configuration"]))
            options = self.parameters._configuration[
                "openpmd_configuration"
            ].copy()
            options["defer_iteration_parsing"] = True
            series = io.Series(
                path, io.Access.read_only, options=json.dumps(options)
            )

            # Check if this actually MALA compatible data.
            if series.get_attribute("is_mala_data") != 1:
                raise Exception(
                    "Non-MALA data detected, cannot work with this data."
                )

            # A bit clanky, but this way only the FIRST iteration is loaded,
            # which is what we need for loading from a single file that
            # may be whatever iteration in its series.
            # Also, in combination with `defer_iteration_parsing`, specified as
            # default above, this opens and parses the first iteration,
            # and no others.
            for current_iteration in series.read_iterations():
                mesh = current_iteration.meshes[self.data_name]
                tuple_from_file = [
                    mesh["0"].shape[0],
                    mesh["0"].shape[1],
                    mesh["0"].shape[2],
                    len(mesh),
                ]
                loaded_dtype = mesh["0"].dtype
                break
            series.close()
        else:
            tuple_from_file = None

        if comm is not None:
            tuple_from_file = comm.bcast(tuple_from_file, root=0)
        if read_dtype:
            return (
                self._process_loaded_dimensions(tuple(tuple_from_file)),
                loaded_dtype,
            )
        else:
            return self._process_loaded_dimensions(tuple(tuple_from_file))

    def write_to_numpy_file(self, path, array):
        """
        Write data to a numpy file.

        Parameters
        ----------
        path : string
            File to save into.

        array : numpy.ndarray
            Array to save.
        """
        np.save(path, array)

    class SkipArrayWriting:
        """
        Optional/alternative parameter for `write_to_openpmd_file` function.

        The function `write_to_openpmd_file` can be used:

        1. either for writing the entire openPMD file, as the name implies
        2. or for preparing and initializing the structure of an openPMD file,
           without actually having the array data at hand yet.
           This means writing attributes, creating the hierarchy and specifying
           the dataset extents and types.

        In the latter case, no numpy array is provided at the call site, but two
        kinds of information are still required for preparing the structure, that
        would normally be extracted from the numpy array:

        1. The dataset extent and type.
        2. The feature size.

        In order to provide this data, the numpy array can be replaced with an
        instance of the class SkipArrayWriting.

        Parameters
        ----------
        dataset : openpmd_api.Dataset
            OpenPMD Data set to eventually write to.

        feature_size : int
            Size of the feature dimension.

        Attributes
        ----------
        dataset : mala.Parameters
            OpenPMD Data set to eventually write to.

        feature_size : list
            Size of the feature dimension.
        """

        # dataset has type openpmd_api.Dataset (not adding a type hint to avoid
        # needing to import here)
        def __init__(self, dataset, feature_size):
            self.dataset = dataset
            self.feature_size = feature_size

    def write_to_openpmd_file(
        self,
        path,
        array,
        additional_attributes={},
        internal_iteration_number=0,
    ):
        """
        Write data to an OpenPMD file.

        Parameters
        ----------
        path : string
            File to save into. If no file ending is given, .h5 is assumed.
            Alternatively: A Series, opened already.

        array : Either numpy.ndarray or an SkipArrayWriting object
            Either the array to save or the meta information needed to create
            the openPMD structure.

        additional_attributes : dict
            Dictionary containing additional attributes to be saved.

        internal_iteration_number : int
            Internal OpenPMD iteration number. Ideally, this number should
            match any number present in the file name, if this data is part
            of a larger data set.
        """
        import openpmd_api as io

        if isinstance(path, str):
            directory, file_name = os.path.split(path)
            path = os.path.join(directory, file_name.replace("*", "%T"))
            file_ending = file_name.split(".")[-1]
            if file_name == file_ending:
                path += ".h5"
            elif file_ending not in io.file_extensions:
                raise Exception("Invalid file ending selected: " + file_ending)
            if self.parameters._configuration["mpi"]:
                series = io.Series(
                    path,
                    io.Access.create,
                    get_comm(),
                    options=json.dumps(
                        self.parameters._configuration["openpmd_configuration"]
                    ),
                )
            else:
                series = io.Series(
                    path,
                    io.Access.create,
                    options=json.dumps(
                        self.parameters._configuration["openpmd_configuration"]
                    ),
                )
        elif isinstance(path, io.Series):
            series = path

        series.set_attribute("is_mala_data", 1)
        series.set_software(name="MALA", version=mala_version)
        series.author = "..."
        for entry in additional_attributes:
            series.set_attribute(entry, additional_attributes[entry])

        iteration = series.write_iterations()[internal_iteration_number]

        # This function may be called without the feature dimension
        # explicitly set (i.e. during testing or post-processing).
        # We have to check for that.
        if self.feature_size == 0 and not isinstance(
            array, self.SkipArrayWriting
        ):
            self._set_feature_size_from_array(array)

        self.write_to_openpmd_iteration(iteration, array)
        return series

    def write_to_openpmd_iteration(
        self,
        iteration,
        array,
        local_offset=None,
        local_reach=None,
        additional_metadata=None,
        feature_from=0,
        feature_to=None,
    ):
        """
        Write a file within an OpenPMD iteration.

        Parameters
        ----------
        iteration : OpenPMD iteration
            OpenPMD iteration into which to save.

        array : numpy.ndarry
            Array to save.

        additional_metadata : list
            If not None, and the selected class implements it, additional
            metadata will be read from this source. This metadata will then,
            depending on the class, be saved in the OpenPMD file.

        local_offset  : list
            [x,y,z] value from which to start writing the array.

        local_reach  : list
            [x,y,z] value until which to read the array.

        feature_from  : int
            Value from which to start writing in the feature dimension. With
            this parameter and feature_to, one can parallelize over the feature
            dimension.

        feature_to : int
            Value until which to write in the feature dimension. With
            this parameter and feature_from, one can parallelize over the feature
            dimension.
        """
        import openpmd_api as io

        if local_offset is None:
            [x_from, y_from, z_from] = [None, None, None]
        else:
            [x_from, y_from, z_from] = local_offset
        if local_reach is None:
            [x_to, y_to, z_to] = [None, None, None]
        else:
            [x_to, y_to, z_to] = local_reach

        mesh = iteration.meshes[self.data_name]

        if additional_metadata is not None:
            self._process_additional_metadata(additional_metadata)
        self._set_openpmd_attribtues(iteration, mesh)

        # If the data contains atomic data, we need to process it.
        atoms_ase = self._get_atoms()
        if atoms_ase is not None:
            # This data is equivalent across the ranks, so just write it once
            atoms_openpmd = iteration.particles["atoms"]
            atomic_positions = atoms_ase.get_positions()
            atomic_numbers = atoms_ase.get_atomic_numbers()
            positions = io.Dataset(
                # Need bugfix https://github.com/openPMD/openPMD-api/pull/1357
                (
                    atomic_positions[0].dtype
                    if io.__version__ >= "0.15.0"
                    else io.Datatype.DOUBLE
                ),
                atomic_positions[0].shape,
            )
            numbers = io.Dataset(atomic_numbers[0].dtype, [1])
            iteration.set_attribute(
                "periodic_boundary_conditions_x", atoms_ase.pbc[0]
            )
            iteration.set_attribute(
                "periodic_boundary_conditions_y", atoms_ase.pbc[1]
            )
            iteration.set_attribute(
                "periodic_boundary_conditions_z", atoms_ase.pbc[2]
            )
            # atoms_openpmd["position"].time_offset = 0.0
            # atoms_openpmd["positionOffset"].time_offset = 0.0
            for atom in range(0, len(atoms_ase)):
                atoms_openpmd["position"][str(atom)].reset_dataset(positions)
                atoms_openpmd["number"][str(atom)].reset_dataset(numbers)
                atoms_openpmd["positionOffset"][str(atom)].reset_dataset(
                    positions
                )

                atoms_openpmd_position = atoms_openpmd["position"][str(atom)]
                atoms_openpmd_number = atoms_openpmd["number"][str(atom)]
                if get_rank() == 0:
                    atoms_openpmd_position.store_chunk(atomic_positions[atom])
                    atoms_openpmd_number.store_chunk(
                        np.array([atomic_numbers[atom]])
                    )
                atoms_openpmd["positionOffset"][str(atom)].make_constant(0)

                # Positions are stored in Angstrom.
                atoms_openpmd["position"][str(atom)].unit_SI = 1.0e-10
                atoms_openpmd["positionOffset"][str(atom)].unit_SI = 1.0e-10

        if any(i == 0 for i in self.grid_dimensions) and not isinstance(
            array, self.SkipArrayWriting
        ):
            self.grid_dimensions = array.shape[0:-1]

        dataset = (
            array.dataset
            if isinstance(array, self.SkipArrayWriting)
            else io.Dataset(array.dtype, self.grid_dimensions)
        )

        # Global feature sizes:
        feature_global_from = 0
        feature_global_to = self.feature_size
        if feature_global_to == 0:
            feature_global_to = (
                array.feature_size
                if isinstance(array, self.SkipArrayWriting)
                else array.shape[-1]
            )

        # First loop: Only metadata, write metadata equivalently across ranks
        for current_feature in range(feature_global_from, feature_global_to):
            mesh_component = mesh[str(current_feature)]
            mesh_component.reset_dataset(dataset)
            # All data is assumed to be saved in
            # MALA units, so the SI conversion factor we save
            # here is the one for MALA (ASE) units
            mesh_component.unit_SI = self.si_unit_conversion
            # position: which relative point within the cell is
            # represented by the stored values
            # ([0.5, 0.5, 0.5] represents the middle)
            mesh_component.position = [0.5, 0.5, 0.5]

        if isinstance(array, self.SkipArrayWriting):
            return

        if feature_to is None:
            feature_to = array.shape[3]

        if feature_to - feature_from != array.shape[3]:
            raise RuntimeError(
                """\
[write_to_openpmd_iteration] Internal error, called function with
wrong parameters. Specification of features ({} - {}) on rank {} does not
match the array dimensions (extent {} in the feature dimension)""".format(
                    feature_from, feature_to, get_rank(), array.shape[3]
                )
            )

        # See above - will currently break for density of states,
        # which is something we never do though anyway.
        # Deal with `granularity` items of the vectors at a time
        # Or in the openPMD layout: with `granularity` record components
        granularity = self.parameters._configuration["openpmd_granularity"]
        # Before writing the actual data, we have to make two considerations
        # for MPI:
        # 1) The following loop does not necessarily have the same number of
        #    iterations across MPI ranks. Since the Series is flushed in every
        #    loop iteration, we need to harmonize the number of flushes
        #    (flushing is collective).
        # 2) Dataset creation and attribute writing is collective in HDF5.
        #    So, we need to define all features on all ranks, even if not all
        #    features are written from all ranks.
        if self.parameters._configuration["mpi"]:
            from mpi4py import MPI

            my_iteration_count = len(range(0, array.shape[3], granularity))
            highest_iteration_count = get_comm().allreduce(
                my_iteration_count, op=MPI.MAX
            )
            extra_flushes = highest_iteration_count - my_iteration_count
        else:
            extra_flushes = 0

        # Second loop: Write heavy data
        for base in range(0, array.shape[3], granularity):
            end = min(base + granularity, array.shape[3])
            transposed = np.transpose(
                array[:, :, :, base:end], axes=[3, 0, 1, 2]
            ).copy()
            for i in range(base, end):
                # i is the index within the array passed to this function.
                # The feature corresponding to this index is offset
                # by the feature_from parameter
                current_feature = i + feature_from
                mesh_component = mesh[str(current_feature)]

                mesh_component[x_from:x_to, y_from:y_to, z_from:z_to] = (
                    transposed[i - base, :, :, :]
                )

            iteration.series_flush()

        # Third loop: Extra flushes to harmonize ranks
        for _ in range(extra_flushes):
            # This following line is a workaround for issue
            # https://github.com/openPMD/openPMD-api/issues/1616
            # Fixed in openPMD-api 0.16 by
            # https://github.com/openPMD/openPMD-api/pull/1619
            iteration.dt = iteration.dt
            iteration.series_flush()

        iteration.close(flush=True)

    ##############################
    # Class-specific reshaping, processing, etc. of data.
    #    Has to be implemented by the classes themselves. E.g. descriptors may
    #    need to cut xyz-coordinates, LDOS/density may need unit conversion.
    ##############################

    @abstractmethod
    def _process_loaded_array(self, array, units=None):
        """
        Process loaded array (i.e., unit change, reshaping, etc.).

        Parameters
        ----------
        array : numpy.ndarray
            Array to process.

        units : string
            Units of input array.
        """
        pass

    @abstractmethod
    def _process_loaded_dimensions(self, array_dimensions):
        """
        Process loaded dimensions.

        E.g., this could include cutting feature dimensions reserved for
        coordinate data in descriptor data.

        Parameters
        ----------
        array_dimensions : tuple
            Raw dimensions of the array.
        """
        pass

    @abstractmethod
    def _set_feature_size_from_array(self, array):
        """
        Set the feature size from the array.

        Feature sizes are saved in different ways for different physical data
        classes.

        Parameters
        ----------
        array : numpy.ndarray
        """
        pass

    def _process_additional_metadata(self, additional_metadata):
        """
        Process additional metadata.

        If additional metadata is provided, and saving is performed via
        openpmd, then this function process the metadata dictionary in a way
        it can be saved by openpmd.

        Parameters
        ----------
        additional_metadata : dict
            Dictionary containing additional metadata.
        """
        pass

    def _feature_mask(self):
        """
        Return a mask for features that are not part of the feature dimension.

        The mask assumes that the features which do not belong to the feature
        dimension are at the beginning of the array.

        Returns
        -------
        mask : int
            Starting index after which the actual feature dimension starts.
        """
        return 0

    def _set_geometry_info(self, mesh):
        """
        Set geometry information to openPMD mesh.

        This has to be done as part of the openPMD saving process.

        Parameters
        ----------
        mesh : openpmd_api.Mesh
            OpenPMD mesh for which to set geometry information.
        """
        pass

    def _set_openpmd_attribtues(self, iteration, mesh):
        """
        Set openPMD attributes.

        This has to be done as part of the openPMD saving process. It saves
        metadata related to the saved data to make data more reproducible.

        Parameters
        ----------
        iteration : openpmd_api.Iteration
            OpenPMD iteration for which to set attributes.

        mesh : openpmd_api.Mesh
            OpenPMD mesh for which to set attributes.
        """
        mesh.unit_dimension = self.si_dimension
        mesh.axis_labels = ["x", "y", "z"]
        mesh.grid_global_offset = [0, 0, 0]

        # MALA internally operates in Angstrom (10^-10 m)
        mesh.grid_unit_SI = 1e-10

        mesh.comment = (
            "This is a special geometry, based on the cartesian geometry."
        )

        # Fill geometry information (if provided)
        self._set_geometry_info(mesh)

    def _process_openpmd_attributes(self, series, iteration, mesh):
        """
        Process loaded openPMD attributes.

        This is done during loading from OpenPMD data. OpenPMD can save
        metadata not contained in the data itself, but in the file. With this
        function, this information can be loaded to relevant MALA classes.

        Parameters
        ----------
        series : openpmd_api.Series
            OpenPMD series from which to load an iteration.

        iteration : openpmd_api.Iteration
            OpenPMD iteration from which to load.

        mesh : openpmd_api.Mesh
            OpenPMD mesh used during loading.
        """
        self._process_geometry_info(mesh)

    def _process_geometry_info(self, mesh):
        """
        Process loaded openPMD geometry information.

        Information on geometry is one of the pieces of metadata that can be
        saved in OpenPMD files. This function processes this information upon
        loading, and saves it to the correct place in the respective MALA
        class.

        Parameters
        ----------
        mesh : openpmd_api.Mesh
            OpenPMD mesh used during loading.
        """
        pass

    # Currently all data we have is atom based.
    # That may not always be the case.
    def _get_atoms(self):
        """
        Access atoms saved in PhysicalData-derived class.

        For any derived class which is atom based (currently, all are), this
        function returns the atoms, which may not be directly accessible as
        an attribute for a variety of reasons.

        Returns
        -------
        atoms : ase.Atoms
            An ASE atoms object holding the associated atoms of this object.
        """
        return None

    @staticmethod
    def _get_attribute_if_attribute_exists(
        iteration, attribute, default_value=None
    ):
        """
        Access an attribute from an openPMD iteration safely.

        If the attribute does not exist, a default values is returned.

        Parameters
        ----------
        iteration : openpmd_api.Iteration
            OpenPMD iteration from which to load an attribute.

        attribute : string
            Name of the attribute to load.

        default_value : any
            Default value to return if the attribute does not exist.

        Returns
        -------
        value : any
            Value of the attribute if it exists, else the default value.
        """
        if attribute in iteration.attributes:
            return iteration.get_attribute(attribute)
        else:
            return default_value
