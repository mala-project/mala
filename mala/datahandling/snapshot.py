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

    selection_mask : None or [boolean]
        If None, entire snapshot is loaded, if [boolean], it is used as a
        mask to select which examples are loaded
    """

    def __init__(self, input_npy_file, input_npy_directory,
                 output_npy_file,  output_npy_directory,
                 snapshot_function,
                 input_units="", output_units="",
                 calculation_output="",
                 snapshot_type="openpmd", selection_mask=None):
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

        # Legacy functionality: Determine whether the snapshot contains
        # numpy or openpmd files.
        self.snapshot_type = snapshot_type

        # All the dimensionalities of the snapshot.
        self.grid_dimensions = None
        self.grid_size = None
        self.input_dimension = None
        self.output_dimension = None
    
        # Mask determining which examples from the snapshot to use
        if isinstance(selection_mask, np.ndarray):
            self._selection_mask = selection_mask.tolist()
        else: 
            self._selection_mask = selection_mask


    def set_selection_mask(self, selection_mask):
        if isinstance(selection_mask, np.ndarray):
            self._selection_mask = selection_mask.tolist()
        else: 
            self._selection_mask = selection_mask
        if selection_mask is not None:
            self.grid_size = sum(self._selection_mask)
        # TODO also adjust other dimensinot params


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
                                  json_dict["snapshot_function"],
                                  json_dict["snapshot_type"],
                                  json_dict["selection_mask"])
        for key in json_dict:
            setattr(deserialized_object, key, json_dict[key])
        return deserialized_object
