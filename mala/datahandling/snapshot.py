"""Snapshot class."""

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
    """

    def __init__(self, input_npy_file="", input_npy_directory="",
                 input_units="",  output_npy_file="",
                 output_npy_directory="", output_units="",
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

    def to_json(self):
        return self._standard_serializer()
