"""Base class for all calculators that deal with physical data"""
from abc import ABC, abstractmethod

import numpy as np
import openpmd_api as io


class PhysicalData(ABC):
    """
    Base class for physical data.

    Implements general framework to read and write such data to and from
    files.
    """

    ##############################
    # Constructors
    ##############################

    def __init__(self, parameters):
        self.parameters = parameters

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
    def si_dimension(self):
        """
        Dictionary containing the SI unit dimensions in OpenPMD format

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

    def read_from_numpy_file(self, path, units=None):
        """
        Read the data from a numpy file.

        Parameters
        ----------
        path : string
            Path to the numpy file.

        units : string
            Units the data is saved in.

        Returns
        -------
        data : numpy.ndarray
            A numpy array containing the data.

        """
        loaded_array = np.load(path)
        return self._process_loaded_array(loaded_array, units=units)

    def read_from_openpmd_file(self, path, units=None):
        """
        Read the data from a numpy file.

        Parameters
        ----------
        path : string
            Path to the openPMD file.

        units : string
            Units the data is saved in.

        Returns
        -------
        data : numpy.ndarray
            A numpy array containing the data.

        """
        series = io.Series(path, io.Access.read_only,
                           options=self.parameters._configuration["openpmd_configuration"])

        # Check if this actually MALA compatible data.
        if series.get_attribute("is_mala_data") != 1:
            raise Exception("Non-MALA data detected, cannot work with this "
                            "data.")

        # A bit clanky, but this way only the FIRST iteration is loaded,
        # which is what we need for loading from a single file that
        # may be whatever iteration in its series.
        for current_iteration in series.read_iterations():
            mesh = current_iteration.meshes[self.data_name]
            break

        # TODO: Are there instances in MALA, where we wouldn't just label
        # the feature dimension with 0,1,... ? I can't think of one.
        # But there may be in the future, and this'll break
        data = np.zeros((mesh["0"].shape[0], mesh["0"].shape[1],
                         mesh["0"].shape[2], len(mesh)), dtype=mesh["0"].dtype)

        # TODO: For Franz, as discussed.
        for i in range(0, len(mesh)):
            temp_descriptors = mesh[str(i)].load_chunk()
            series.flush()
            data[:, :, :, i] = temp_descriptors.copy()

        return self._process_loaded_array(data, units=units)

    def read_dimensions_from_numpy_file(self, path):
        """
        Read only the dimensions from a numpy file.

        Parameters
        ----------
        path : string
            Path to the numpy file.
        """
        loaded_array = np.load(path, mmap_mode="r")
        return self._process_loaded_dimensions(np.shape(loaded_array))

    def read_dimensions_from_openpmd_file(self, path):
        """
        Read only the dimensions from a openPMD file.

        Parameters
        ----------
        path : string
            Path to the openPMD file.
        """
        series = io.Series(path, io.Access.read_only,
                           options=self.parameters._configuration["openpmd_configuration"])

        # Check if this actually MALA compatible data.
        if series.get_attribute("is_mala_data") != 1:
            raise Exception("Non-MALA data detected, cannot work with this "
                            "data.")

        # A bit clanky, but this way only the FIRST iteration is loaded,
        # which is what we need for loading from a single file that
        # may be whatever iteration in its series.
        for current_iteration in series.read_iterations():
            mesh = current_iteration.meshes[self.data_name]
            return self.\
                _process_loaded_dimensions((mesh["0"].shape[0],
                                            mesh["0"].shape[1],
                                            mesh["0"].shape[2],
                                            len(mesh)))

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

    def write_to_openpmd_iteration(self, iteration, array):
        """
        Write a file within an OpenPMD iteration.

        Parameters
        ----------
        iteration : OpenPMD iteration
            OpenPMD iteration into which to save.

        array : numpy.ndarry
            Array to save.

        """
        mesh = iteration.meshes[self.data_name]
        self._set_openpmd_attribtues(mesh)
        dataset = io.Dataset(array.dtype,
                             array[:, :, :, 0].shape)

        # See above - will currently break for density of states,
        # which is something we never do though anyway.
        # Deal with `granularity` items of the vectors at a time
        # Or in the openPMD layout: with `granularity` record components
        granularity = 16 # just some random value for now
        for base in range(0, array.shape[3], granularity):
            end = min(base + granularity, array.shape[3])
            transposed = \
                np.transpose(array[:, :, :, base:end], axes=[3, 0, 1, 2]).copy()
            for i in range(base, end):
                mesh_component = mesh[str(i)]
                mesh_component.reset_dataset(dataset)

                # mesh_component[:, :, :] = transposed[i - base, :, :, :]
                mesh_component.store_chunk(transposed[i - base, :, :, :])

                # All data is assumed to be saved in
                # MALA units, so the SI conversion factor we save
                # here is the one for MALA (ASE) units
                mesh_component.unit_SI = self.si_unit_conversion
                # position: which relative point within the cell is
                # represented by the stored values
                # ([0.5, 0.5, 0.5] represents the middle)
                mesh_component.position = [0.5, 0.5, 0.5]
            iteration.series_flush()

        iteration.close(flush=True)

    ##############################
    # Class-specific reshaping, processing, etc. of data.
    #    Has to be implemented by the classes themselves. E.g. descriptors may
    #    need to cut xyz-coordinates, LDOS/density may need unit conversion.
    ##############################

    @abstractmethod
    def _process_loaded_array(self, array, units=None):
        pass

    @abstractmethod
    def _process_loaded_dimensions(self, array_dimensions):
        pass

    def _set_openpmd_attribtues(self, mesh):
        mesh.unit_dimension = self.si_dimension
        mesh.axis_labels = ["x", "y", "z"]
        mesh.grid_global_offset = [0, 0, 0]
        mesh.grid_spacing = [1, 1, 1]
        mesh.grid_unit_SI = 1

        # for specifying one of the standardized geometries
        mesh.geometry = io.Geometry.cartesian
        # or for specifying a custom one
        mesh.geometry = io.Geometry.other
        # only supported on dev branch so far
        # input_mesh.geometry = "other:my_geometry"
        # custom geometries might need further
        #  custom information
        mesh.set_attribute("angles", [45, 90, 90])
        # set a comment that will appear in the dataset on-disk
        mesh.comment = \
            "This is a special geometry, " \
            "based on the cartesian geometry."

