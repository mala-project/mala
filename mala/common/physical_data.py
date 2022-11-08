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

    def __init__(self):
        pass

    ##############################
    # Properties
    ##############################

    @property
    @abstractmethod
    def data_name(self):
        """Get a string that describes the data (for e.g. metadata)."""
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
        series = io.Series(path, io.Access.read_only)

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

        return self._process_loaded_array(data)

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
        series = io.Series(path, io.Access.read_only)

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
