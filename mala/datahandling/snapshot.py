"""Represents an entire atomic snapshot (including descriptor/target data)."""
from os.path import join

import numpy as np

from mala.common.json_serializable import JSONSerializable


class Snapshot(JSONSerializable):
    """
    Represents a snapshot on a hard drive.

    A snapshot consists of numpy arrays for input/output data and an
    optional DFT calculation output, needed for post-processing.

    Parameters
    ----------
    input_npy_file : string
        File with saved numpy input array.

    input_npy_directory : string
        Directory containing input_npy_directory.

    output_npy_file : string
        File with saved numpy output array.

    output_npy_directory : string
        Directory containing output_npy_file.

    input_units : string
        Units of input data. See descriptor classes to see which units are
        supported.

    output_units : string
        Units of output data. See target classes to see which units are
        supported.

    calculation_output : string
        File with the output of the original snapshot calculation. This is
        only needed when testing multiple snapshots.

    snapshot_function : string
        "Function" of the snapshot in the MALA workflow.

          - te: This snapshot will be a testing snapshot.
          - tr: This snapshot will be a training snapshot.
          - va: This snapshot will be a validation snapshot.

        Replaces the old approach of MALA to have a separate list.
        Default is None.
    """

    def __init__(self, input_npy_file, input_npy_directory,
                 output_npy_file,  output_npy_directory,
                 snapshot_function,
                 input_units="", output_units="",
                 calculation_output=""):
        super(Snapshot, self).__init__()

        # Inputs.
        self.input_npy_file = input_npy_file
        self.input_npy_directory = input_npy_directory
        self.input_units = input_units

        # Outputs.
        self.output_npy_file = output_npy_file
        self.output_npy_directory = output_npy_directory
        self.output_units = output_units

        # Calculation output.
        self.calculation_output = calculation_output

        # Function of the snapshot.
        self.snapshot_function = snapshot_function

        # All the dimensionalities of the snapshot.
        self.grid_dimensions = None
        self.grid_size = None
        self.input_dimension = None
        self.output_dimension = None

    def load_dimensions(self, descriptors_contain_xyz,
                        debug_dimensions=None):
        """
        Load the dimensions for a snapshot from the linked files.

        Parameters
        ----------
        descriptors_contain_xyz :
            If True, the first 3 entries in the feature dimension are
            assumed to be xyz-coordinates and will be ignored for
            the calculation of the feature dimension.

        debug_dimensions :
            If not None, these dimensions will be used as xyz-dimensions.
            Useful for debugging.

        """
        # Load input and output data separately and see if they match.

        # Input data.
        file = join(self.input_npy_directory, self.input_npy_file)
        loaded_array = np.load(file, mmap_mode="r")
        if debug_dimensions is not None:
            if len(debug_dimensions) == 3:
                loaded_array = loaded_array[0:debug_dimensions[0],
                                            0:debug_dimensions[1],
                                            0:debug_dimensions[2], :]
        input_dimensions = np.shape(loaded_array)[0:3]
        self.input_dimension = np.shape(loaded_array)[3]
        if descriptors_contain_xyz:
            self.input_dimension -= 3

        # Output data
        file = join(self.output_npy_directory, self.output_npy_file)
        loaded_array = np.load(file, mmap_mode="r")
        if debug_dimensions is not None:
            if len(debug_dimensions) == 3:
                loaded_array = loaded_array[0:debug_dimensions[0],
                                            0:debug_dimensions[1],
                                            0:debug_dimensions[2], :]
        output_dimensions = np.shape(loaded_array)[0:3]
        self.output_dimension = np.shape(loaded_array)[3]

        if input_dimensions[0] != output_dimensions[0] \
                or input_dimensions[1] != output_dimensions[1] \
                or input_dimensions[2] != output_dimensions[2]:
            return False
        else:
            self.grid_dimensions = input_dimensions
            self.grid_size = int(np.prod(self.grid_dimensions))
            return True

    @classmethod
    def from_json(cls, json_dict):
        """
        Read this object from a dictionary saved in a JSON file.

        Parameters
        ----------
        json_dict : dict
            A dictionary containing all attributes, properties, etc. as saved
            in the json file.

        Returns
        -------
        deserialized_object : JSONSerializable
            The object as read from the JSON file.

        """
        deserialized_object = cls(json_dict["input_npy_file"],
                                  json_dict["input_npy_directory"],
                                  json_dict["output_npy_file"],
                                  json_dict["output_npy_directory"],
                                  json_dict["snapshot_function"])
        for key in json_dict:
            setattr(deserialized_object, key, json_dict[key])
        return deserialized_object
